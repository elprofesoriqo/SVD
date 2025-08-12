import os
import argparse
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, ProfilerActivity

# ============== Complex helpers ==============

def to_complex(x: Tensor) -> Tensor:
    return torch.view_as_complex(x.contiguous())

def from_complex(z: Tensor) -> Tensor:
    return torch.view_as_real(z).contiguous()

def column_normalize(Z: Tensor, eps: float = 1e-8) -> Tensor:
    # Z: (B, L, r)
    n = Z.abs().pow(2).sum(dim=1, keepdim=True).clamp_min(eps).sqrt()
    return Z / n

# ============== Model (Round2, U=V) ==============

class UVNetFFT_UeqV(nn.Module):
    """
    Lekkie kodowanie: flatten(X) + |FFT2(X)| -> MLP -> U (complex, unit-norm) + wstępne s.
    V nie jest uczone; przyjmujemy U=V po korekcji faz.
    """
    def __init__(self, M: int, N: int, r: int, width: int = 96):
        super().__init__()
        self.M, self.N, self.r = M, N, r
        d = M*N
        self.enc_x = nn.Sequential(
            nn.Linear(d*2, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
        )
        self.enc_fft = nn.Sequential(
            nn.Linear(d, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
        )
        self.mix = nn.Sequential(
            nn.Linear(width, width), nn.GELU(),
        )
        self.u_r = nn.Linear(width, M*r)
        self.u_i = nn.Linear(width, M*r)
        self.s_head = nn.Linear(width, r)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x: (B,M,N,2)
        B, M, N, C = x.shape
        assert C == 2 and M == self.M and N == self.N
        z  = x.reshape(B, -1)              # (B, 2MN)
        Hn = to_complex(x)                 # (B,M,N)
        F2 = torch.fft.fft2(Hn, norm="ortho")
        aF = torch.abs(F2).reshape(B, -1)  # (B, MN)

        hx = self.enc_x(z)
        hf = self.enc_fft(aF)
        h  = 0.5*(hx + hf)
        h  = self.mix(h)

        Ur = self.u_r(h).view(B, M, self.r)
        Ui = self.u_i(h).view(B, M, self.r)
        U0 = torch.complex(Ur, Ui)
        U0 = column_normalize(U0)

        s0 = F.softplus(self.s_head(h)) + 1e-8
        return U0, s0

class SRefiner(nn.Module):
    """
    Prosty refiner s bazujący na G = U^H H U (Re/Im).
    """
    def __init__(self, r: int, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*r*r, hidden), nn.GELU(),
            nn.Linear(hidden, r),
        )
    def forward(self, G: Tensor, s_raw: Tensor) -> Tensor:
        B, r, _ = G.shape
        feat = from_complex(G).reshape(B, -1)
        delta = self.net(feat)
        return F.softplus(s_raw + delta) + 1e-8

# ============== U=V helpers, sorting, losses ==============

def phase_align_columns(G: Tensor) -> Tensor:
    # G = U^H H U; zwraca D, aby diag(D^H G) była dodatnia (korekcja faz)
    d = torch.diagonal(G, dim1=-2, dim2=-1)
    phi = torch.angle(d)
    D = torch.diag_embed(torch.polar(torch.ones_like(phi), -phi))
    return D

def sort_by_desc(s: Tensor, U: Tensor) -> Tuple[Tensor, Tensor]:
    idx = torch.argsort(s, dim=-1, descending=True)
    B, M, r = U.shape
    s2 = s.gather(-1, idx)
    idxU = idx.unsqueeze(1).expand(B, M, r)
    U2 = U.gather(-1, idxU)
    return s2, U2

def apply_rank_mask(U: Tensor, s: Tensor, active_r: int) -> Tuple[Tensor, Tensor]:
    if active_r >= s.shape[-1]:
        return U, s
    mask = torch.zeros_like(s)
    mask[:, :active_r] = 1.0
    return U * mask.unsqueeze(1), s * mask

def mono_penalty(s: Tensor) -> Tensor:
    if s.shape[1] <= 1: return s.mean()*0
    return F.relu(s[:,1:] - s[:,:-1]).mean()

def loss_Round2(Y: Tensor, Xn: Tensor, U: Tensor, s: Tensor,
                lam_off: float, lam_ort: float):
    """
    AE dla Round2 (U=V):
      rec  = ||Y - U S U^H|| / ||Y||
      ort  = ||U^H U - I||
      off  = ||offdiag(U^H X U)|| / ||diag(U^H X U)||
    """
    Hl = to_complex(Y)
    Hhat = (U * s.unsqueeze(-2)) @ U.conj().transpose(-2, -1)
    rec = (Hl - Hhat).norm(dim=(-2, -1)) / (Hl.norm(dim=(-2, -1)) + 1e-8)

    I = torch.eye(U.shape[-1], device=U.device, dtype=U.dtype).unsqueeze(0)
    ort = ((U.conj().transpose(-2, -1) @ U) - I).norm(dim=(-2, -1))

    Gx = U.conj().transpose(-2, -1) @ Xn @ U
    D  = torch.diag_embed(torch.diagonal(Gx, dim1=-2, dim2=-1))
    scale = D.norm(dim=(-2, -1)).clamp_min(1e-8)
    off = (Gx - D).norm(dim=(-2, -1)) / scale

    L = rec.mean() + lam_ort*ort.mean() + lam_off*off.mean()
    return L, rec.mean(), ort.mean(), off.mean()

# ============== Dataset (Round2) with RECOMPUTE CACHE ==============

class Round2Dataset(Dataset):
    """
    Round2 scenariusze 1..3.
    - Ładuje TrainData/TrainLabel.
    - Skaluje próbki przez ||X||_F (spójne z inferencją).
    - (RECOMPUTE) Wyznacza i cache'uje SVD etykiet:
        domyślnie tylko S -> Round2S{k}.npy
        opcjonalnie także U,V -> Round2U{k}.npy, Round2V{k}.npy
    """
    def __init__(self, data_dir: str, r_core: int,
                 cache_uv: bool = False, recompute: bool = False):
        super().__init__()
        self.r = r_core
        self.cache_uv = cache_uv

        self.xs: List[np.ndarray] = []
        self.ys: List[np.ndarray] = []
        self.Ss: List[np.ndarray] = []

        self.M = self.N = None

        cfg = {}
        for k in (1, 2, 3, 4):
            cfgp = os.path.join(data_dir, f"Round2CfgData{k}.txt")
            tdp  = os.path.join(data_dir, f"Round2TrainData{k}.npy")
            tlp  = os.path.join(data_dir, f"Round2TrainLabel{k}.npy")
            if os.path.exists(cfgp) and os.path.exists(tdp) and os.path.exists(tlp):
                lines = [l.strip() for l in open(cfgp, "r", encoding="utf-8") if l.strip()]
                M = int(''.join(filter(str.isdigit, lines[1])))
                N = int(''.join(filter(str.isdigit, lines[2])))
                cfg[k] = (M, N)

        if not cfg:
            raise FileNotFoundError("Brak plików Round2* w katalogu data_dir.")

        k0 = list(cfg.keys())[0]
        self.M, self.N = cfg[k0]

        for k, (M, N) in cfg.items():
            tdp = os.path.join(data_dir, f"Round2TrainData{k}.npy")
            tlp = os.path.join(data_dir, f"Round2TrainLabel{k}.npy")
            Xall = np.load(tdp, mmap_mode="r")
            Yall = np.load(tlp, mmap_mode="r")

            # ścieżki cache
            up = os.path.join(data_dir, f"Round2U{k}.npy")
            sp = os.path.join(data_dir, f"Round2S{k}.npy")
            vp = os.path.join(data_dir, f"Round2V{k}.npy")

            need_recompute = recompute
            if not need_recompute:
                # sprawdź dostępność cache
                if self.cache_uv:
                    need_recompute = not (os.path.exists(up) and os.path.exists(sp) and os.path.exists(vp))
                else:
                    need_recompute = not os.path.exists(sp)

                # sprawdź zgodność rozmiaru r
                if not need_recompute and os.path.exists(sp):
                    try:
                        Sprobe = np.load(sp, mmap_mode="r")
                        if Sprobe.shape[1] != self.r:
                            need_recompute = True
                    except Exception:
                        need_recompute = True

            if need_recompute:
                # licz SVD na labelach (poza siecią) i cache'uj
                Uall = None
                Vall = None
                if self.cache_uv:
                    Uall = np.zeros((Yall.shape[0], M, self.r), dtype=np.complex64)
                    Vall = np.zeros((Yall.shape[0], N, self.r), dtype=np.complex64)
                Sall = np.zeros((Yall.shape[0], self.r), dtype=np.float32)

                for i in range(Yall.shape[0]):
                    Yc = Yall[i, ..., 0] + 1j*Yall[i, ..., 1]
                    U, S, Vh = np.linalg.svd(Yc, full_matrices=False)
                    if self.cache_uv:
                        Uall[i] = U[:, :self.r]
                        Vall[i] = Vh.conj().T[:, :self.r]
                    Sall[i] = S[:self.r].astype(np.float32)

                if self.cache_uv:
                    np.save(up, Uall)
                    np.save(vp, Vall)
                np.save(sp, Sall)

            # wczytaj cache S
            Sall = np.load(sp, mmap_mode="r")
            for i in range(Yall.shape[0]):
                self.xs.append(Xall[i])
                self.ys.append(Yall[i])
                self.Ss.append(Sall[i])

        self.n = len(self.xs)

    def __len__(self): return self.n

    def __getitem__(self, idx: int):
        X = torch.from_numpy(self.xs[idx].copy()).float()  # (M,N,2)
        Y = torch.from_numpy(self.ys[idx].copy()).float()
        S = torch.from_numpy(np.array(self.Ss[idx]).copy()).float()  # (r,)
        return X, Y, S

def collate_fn(batch):
    X, Y, S = zip(*batch)
    X = torch.stack(X)
    Y = torch.stack(Y)
    S = torch.stack(S)
    Hx = to_complex(X)
    scale_mat = Hx.norm(dim=(-2, -1), keepdim=True).unsqueeze(-1).clamp_min(1e-8)  # (B,1,1,1)
    scale_s   = Hx.norm(dim=(-2, -1)).clamp_min(1e-8).reshape(-1, 1)               # (B,1)
    return X/scale_mat, Y/scale_mat, S/scale_s

# ============== FLOPs + export ==============

class CombinedForFlops(nn.Module):
    def __init__(self, core: UVNetFFT_UeqV, sref: SRefiner):
        super().__init__()
        self.core = core
        self.sref = sref
    def forward(self, x: Tensor) -> Tensor:
        Hn = to_complex(x)
        U0, s0 = self.core(x)
        G  = U0.conj().transpose(-2, -1) @ Hn @ U0
        s1 = self.sref(G, s0)
        return s1

def get_avg_flops(model: nn.Module, input_data: Tensor) -> float:
    if input_data.dim() == 0 or input_data.size(0) == 0:
        raise RuntimeError("Input data must have a non-zero batch dimension")
    bs = input_data.size(0)
    m = model.eval().cpu()
    inp = input_data.cpu()
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], with_flops=True, record_shapes=False) as prof:
            m(inp)
    total = 0
    for e in prof.events():
        if hasattr(e, "flops") and e.flops is not None:
            total += e.flops
    return (total/bs) * 1e-6 / 2.0

def read_cfg(cfg_path: str):
    lines = [l.strip() for l in open(cfg_path, 'r', encoding='utf-8') if l.strip()]
    M = int(''.join(filter(str.isdigit, lines[1])))
    N = int(''.join(filter(str.isdigit, lines[2])))
    r_full = int(''.join(filter(str.isdigit, lines[4])))
    return M, N, r_full

@torch.no_grad()
def export_npz(core: UVNetFFT_UeqV, sref: SRefiner, test_path: str, out_path: str, device, C: float):
    X = np.load(test_path, mmap_mode='r')
    bs = 256
    U_list, S_list, V_list = [], [], []
    for i in range(0, X.shape[0], bs):
        xb = torch.from_numpy(X[i:i+bs].copy()).float().to(device)
        H  = to_complex(xb)
        scale = H.norm(dim=(-2, -1), keepdim=True).unsqueeze(-1).clamp_min(1e-8)
        xb_n = xb / scale

        # forward
        U0, s0 = core(xb_n)
        Hn = to_complex(xb_n)
        G  = U0.conj().transpose(-2, -1) @ Hn @ U0
        s1 = sref(G, s0)

        # sort + fazy
        s1, U1 = sort_by_desc(s1, U0)
        Gs = U1.conj().transpose(-2, -1) @ Hn @ U1
        D  = phase_align_columns(Gs)
        V1 = U1 @ D  # U=V po korekcji faz

        # odskaluj s
        s_un = s1 * H.norm(dim=(-2, -1)).reshape(-1, 1)

        U_list.append(from_complex(U1).cpu().numpy())
        S_list.append(s_un.cpu().numpy())
        V_list.append(from_complex(V1).cpu().numpy())

    U = np.concatenate(U_list, axis=0)
    S = np.concatenate(S_list, axis=0)
    V = np.concatenate(V_list, axis=0)
    np.savez(out_path, U=U, S=S, V=V, C=float(C))

# ============== Train (Round2 only) ==============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="../../CompetitionData2")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_interval", type=int, default=200)
    # straty
    ap.add_argument("--lambda_off_start", type=float, default=0.10)
    ap.add_argument("--lambda_off_end", type=float, default=0.40)
    ap.add_argument("--lambda_ort", type=float, default=0.5)
    ap.add_argument("--lambda_diag", type=float, default=0.05)
    ap.add_argument("--lambda_mono", type=float, default=2e-3)
    ap.add_argument("--lambda_sval", type=float, default=1e-2)
    # ranga i curriculum
    ap.add_argument("--r_core", type=int, default=64)
    ap.add_argument("--ramp_frac", type=float, default=0.40)
    ap.add_argument("--r_warm", type=int, default=12)
    # cache
    ap.add_argument("--cache_uv", type=int, default=0, help="0: cache only S, 1: cache U,S,V")
    ap.add_argument("--recompute", type=int, default=1, help="1: przelicz i nadpisz cache")
    # io
    ap.add_argument("--model_pth", type=str, default="best_r2.pth")
    ap.add_argument("--export_dir", type=str, default="npz_export")
    ap.add_argument("--data_pct", type=float, default=100.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- data (Round2 + recompute cache) ---
    ds = Round2Dataset(args.data_dir, args.r_core,
                       cache_uv=bool(args.cache_uv), recompute=bool(args.recompute))
    n_total = len(ds)
    n_subset = max(1, int(n_total * args.data_pct / 100.0))
    idx = torch.randperm(n_total)
    val_n = max(1, min(n_subset-1, int(0.10 * n_subset)))
    val_idx = idx[:val_n]
    tr_idx  = idx[val_n:n_subset]

    class Subset(Dataset):
        def __init__(self, base, ids): self.b, self.ids = base, ids
        def __len__(self): return len(self.ids)
        def __getitem__(self, i): return self.b[self.ids[i].item()]

    tr_loader = DataLoader(Subset(ds, tr_idx), batch_size=args.batch_size,
                           shuffle=True, drop_last=True, collate_fn=collate_fn, num_workers=0)
    va_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate_fn, num_workers=0)

    M, N, r = ds.M, ds.N, args.r_core
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = UVNetFFT_UeqV(M, N, r, width=args.width).to(device)
    sref = SRefiner(r, hidden=args.width).to(device)

    opt = torch.optim.AdamW(list(core.parameters()) + list(sref.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # FLOPs (UWAGA: to przerzuca moduły na CPU – od razu wracamy na device)
    dummy = torch.randn(1, M, N, 2)
    start_mmacs = get_avg_flops(CombinedForFlops(core, sref), dummy)
    print(f"[{time.strftime('%H:%M:%S')}] MMACs at start: {start_mmacs:.2f}")
    # >>> KLUCZOWA POPRAWKA <<<
    core.to(device)
    sref.to(device)

    best = float("inf")
    ramp_epochs = max(1, int(args.epochs * args.ramp_frac))

    for ep in range(1, args.epochs+1):
        core.train(); sref.train()
        t = (ep - 1) / max(1, args.epochs - 1)
        lam_off = args.lambda_off_start + (args.lambda_off_end - args.lambda_off_start) * t
        lam_ort = args.lambda_ort
        if ep <= ramp_epochs:
            active_r = int(args.r_warm + (r - args.r_warm) * (ep - 1) / max(1, ramp_epochs - 1))
        else:
            active_r = r

        acc = {"rec":0.0,"ort":0.0,"off":0.0,"diag":0.0,"sval":0.0}
        seen = 0

        for bi, (X, Y, Svt) in enumerate(tr_loader, 1):
            X, Y, Svt = X.to(device), Y.to(device), Svt.to(device)
            Xn = to_complex(X)

            U0, s0 = core(X)
            G  = U0.conj().transpose(-2, -1) @ Xn @ U0
            s1 = sref(G, s0)

            s1, U1 = sort_by_desc(s1, U0)
            U1m, s1m = apply_rank_mask(U1, s1, active_r)

            # główna AE
            L_ae, rec, ort, off = loss_Round2(Y, Xn, U1m, s1m, lam_off=lam_off, lam_ort=lam_ort)

            # zgodność diagonali (na Y)
            Gy = U1m.conj().transpose(-2, -1) @ to_complex(Y) @ U1m
            dh = torch.abs(torch.diagonal(Gy, dim1=-2, dim2=-1))  # (B, r_act)
            L_diag = args.lambda_diag * F.smooth_l1_loss(torch.log(s1m + 1e-8),
                                                         torch.log(dh  + 1e-8))

            # monotoniczność
            L_mono = args.lambda_mono * mono_penalty(s1)

            # teacher (SVD z labeli) – tylko pierwsze r_act
            L_s = args.lambda_sval * F.smooth_l1_loss(torch.log(s1[:, :active_r] + 1e-8),
                                                      torch.log(Svt[:, :active_r] + 1e-8))

            L = L_ae + L_diag + L_mono + L_s

            opt.zero_grad(set_to_none=True)
            L.backward()
            nn.utils.clip_grad_norm_(list(core.parameters()) + list(sref.parameters()), 5.0)
            opt.step()

            bs = X.size(0)
            acc["rec"]  += rec.item() * bs
            acc["ort"]  += ort.item() * bs
            acc["off"]  += off.item() * bs
            acc["diag"] += L_diag.item() * bs
            acc["sval"] += L_s.item() * bs
            seen += bs

            if (bi % args.log_interval == 0) or (bi == len(tr_loader)):
                print(f"[{time.strftime('%H:%M:%S')}] ep{ep:02d} {bi}/{len(tr_loader)} "
                      f"rec={acc['rec']/seen:.4f} ort={acc['ort']/seen:.4f} off={acc['off']/seen:.4f} "
                      f"diag={acc['diag']/seen:.4f} sval={acc['sval']/seen:.4f} r_act={active_r} L={L.item():.4f}")

        sch.step()

        # walidacja: AE ≈ rec + ort
        core.eval(); sref.eval()
        vsum = osum = cnt = 0.0
        with torch.no_grad():
            for X, Y, _ in va_loader:
                X, Y = X.to(device), Y.to(device)
                Xn = to_complex(X)
                U0, s0 = core(X)
                G  = U0.conj().transpose(-2, -1) @ Xn @ U0
                s1 = sref(G, s0)
                s1, U1 = sort_by_desc(s1, U0)

                Hl = to_complex(Y)
                Hhat = (U1 * s1.unsqueeze(-2)) @ U1.conj().transpose(-2, -1)
                rec = (Hl - Hhat).norm(dim=(-2, -1)) / (Hl.norm(dim=(-2, -1)) + 1e-8)
                I = torch.eye(U1.shape[-1], device=U1.device, dtype=U1.dtype).unsqueeze(0)
                ort = ((U1.conj().transpose(-2, -1) @ U1) - I).norm(dim=(-2, -1))

                bs = X.size(0)
                vsum += rec.mean().item() * bs
                osum += ort.mean().item() * bs
                cnt  += bs
        AE = (vsum + osum) / max(1, cnt)
        print(f"[{time.strftime('%H:%M:%S')}] VAL ep{ep:02d} AE={AE:.6f}")

        if AE < best:
            best = AE
            blob = {
                "arch": {"M": M, "N": N, "r": r, "width": args.width},
                "core": core.state_dict(),
                "sref": sref.state_dict(),
            }
            torch.save(blob, args.model_pth)

    # --- export Round2 (1..3) ---
    os.makedirs(args.export_dir, exist_ok=True)

    ckpt = torch.load(args.model_pth, map_location="cpu")
    arch = ckpt["arch"]
    core2 = UVNetFFT_UeqV(arch["M"], arch["N"], arch["r"], width=arch["width"])
    sref2 = SRefiner(arch["r"], hidden=arch["width"])
    core2.load_state_dict(ckpt["core"], strict=True)
    sref2.load_state_dict(ckpt["sref"], strict=True)
    core2.eval(); sref2.eval()

    C = get_avg_flops(CombinedForFlops(core2, sref2), torch.randn(1, M, N, 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core2.to(device); sref2.to(device)

    for k in (1, 2, 3, 4):
        cfgp = os.path.join(args.data_dir, f"Round2CfgData{k}.txt")
        testp = os.path.join(args.data_dir, f"Round2TestData{k}.npy")
        if not (os.path.exists(cfgp) and os.path.exists(testp)):
            continue
        Mm, Nn, r_full = read_cfg(cfgp)
        assert Mm == M and Nn == N and r_full == r, "CFG mismatch vs model."
        outp = os.path.join(args.export_dir, f"R2_{k}.npz")
        export_npz(core2, sref2, testp, outp, device, C)
        print(f"[{time.strftime('%H:%M:%S')}] Saved {outp}")

if __name__ == "__main__":
    main()
