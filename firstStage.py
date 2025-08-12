import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, ProfilerActivity
from torch import Tensor

def to_complex(x):
    return torch.view_as_complex(x.contiguous())

def from_complex(z):
    return torch.view_as_real(z).contiguous()

def column_normalize(Z):
    mag = Z.abs().norm(dim=1, keepdim=True).clamp_min(1e-8)
    return Z / mag

class UVNetFFT(nn.Module):
    def __init__(self, M, N, r, width=128):
        super().__init__()
        self.M, self.N, self.r = M, N, r
        d = M * N
        self.enc_x = nn.Sequential(
            nn.Linear(d*2, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU()
        )
        self.enc_fft = nn.Sequential(
            nn.Linear(d, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU()
        )
        self.u_head = nn.Linear(width, M*r*2)
        self.v_head = nn.Linear(width, N*r*2)

    def forward(self, x):
        B, M, N, C = x.shape
        assert C == 2 and M == self.M and N == self.N
        z = x.reshape(B, -1)
        Hc = to_complex(x)
        F2 = torch.fft.fft2(Hc, norm="ortho")
        aF = torch.abs(F2).reshape(B, -1)
        hx = self.enc_x(z)
        hf = self.enc_fft(aF)
        h = 0.5*(hx + hf)
        u = self.u_head(h).reshape(B, M, self.r, 2)
        v = self.v_head(h).reshape(B, N, self.r, 2)
        U0 = to_complex(u)
        V0 = to_complex(v)
        U0 = column_normalize(U0)
        V0 = column_normalize(V0)
        return U0, V0

def phase_align(U, H, V):
    G = U.conj().transpose(-2,-1) @ H @ V
    d = torch.diagonal(G, dim1=-2, dim2=-1)
    phi = torch.angle(d)
    D = torch.diag_embed(torch.polar(torch.ones_like(phi), -phi))
    return U, V @ D

def rayleigh_s(H, U, V):
    G = U.conj().transpose(-2,-1) @ H @ V
    s0 = torch.abs(torch.diagonal(G, dim1=-2, dim2=-1))
    return s0, G

class SRefiner(nn.Module):
    def __init__(self, r, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*r*r, hidden), nn.GELU(),
            nn.Linear(hidden, r)
        )
    def forward(self, G, s_raw):
        B = G.shape[0]
        feat = from_complex(G).reshape(B, -1)
        delta = self.net(feat)
        return F.softplus(s_raw + delta) + 1e-8

def sort_by_desc(s, U, V):
    idx = torch.argsort(s, dim=-1, descending=True)
    B, M, r = U.shape
    N = V.shape[1]
    idxU = idx.unsqueeze(1).expand(B, M, r)
    idxV = idx.unsqueeze(1).expand(B, N, r)
    return s.gather(-1, idx), U.gather(-1, idxU), V.gather(-1, idxV)

def mono_penalty(s):
    return F.relu(s[:,1:] - s[:,-s.shape[1]+0:-1]).mean() if s.shape[1] > 1 else s.mean()*0

def loss_ae(Y, Hn, U, s, V, lam_off):
    Hl = to_complex(Y)
    Hhat = (U * s.unsqueeze(-2)) @ V.conj().transpose(-2,-1)
    rec = (Hl - Hhat).norm(dim=(-2,-1)) / (Hl.norm(dim=(-2,-1)) + 1e-8)
    Iu = torch.eye(U.shape[-1], device=U.device, dtype=U.dtype).unsqueeze(0)
    Iv = torch.eye(V.shape[-1], device=V.device, dtype=V.dtype).unsqueeze(0)
    ort_u = ((U.conj().transpose(-2,-1) @ U) - Iu).norm(dim=(-2,-1))
    ort_v = ((V.conj().transpose(-2,-1) @ V) - Iv).norm(dim=(-2,-1))
    G = U.conj().transpose(-2,-1) @ Hn @ V
    D = torch.diag_embed(torch.diagonal(G, dim1=-2, dim2=-1))
    off = (G - D).norm(dim=(-2,-1))
    L = rec.mean() + ort_u.mean() + ort_v.mean() + lam_off * off.mean()
    return L, rec.mean(), ort_u.mean()+ort_v.mean(), off.mean()

class SVDDataset(Dataset):
    def __init__(self, data_dir, r_core):
        self.xs, self.ys, self.Us, self.Ss, self.Vs = [], [], [], [], []
        cfg = {}
        for k in (1, 2, 3):
            lines = [l.strip() for l in open(os.path.join(data_dir, f"Round1CfgData{k}.txt")) if l.strip()]
            M = int(''.join(filter(str.isdigit, lines[1])))
            N = int(''.join(filter(str.isdigit, lines[2])))
            cfg[k] = (M, N)
        for k, (M, N) in cfg.items():
            td = np.load(os.path.join(data_dir, f"Round1TrainData{k}.npy"), mmap_mode="r")
            tl = np.load(os.path.join(data_dir, f"Round1TrainLabel{k}.npy"), mmap_mode="r")
            u_path = os.path.join(data_dir, f"Round1U{k}.npy")
            s_path = os.path.join(data_dir, f"Round1S{k}.npy")
            v_path = os.path.join(data_dir, f"Round1V{k}.npy")
            recompute = True
            if os.path.exists(u_path) and os.path.exists(s_path) and os.path.exists(v_path):
                Uall = np.load(u_path)
                Sall = np.load(s_path)
                Vall = np.load(v_path)
                if Uall.shape[2] == r_core:
                    recompute = False
            if recompute:
                Uall = np.zeros((tl.shape[0], M, r_core), dtype=np.complex64)
                Sall = np.zeros((tl.shape[0], r_core), dtype=np.float32)
                Vall = np.zeros((tl.shape[0], N, r_core), dtype=np.complex64)
                for i in range(tl.shape[0]):
                    Yc = tl[i][..., 0] + 1j * tl[i][..., 1]
                    U, S, Vh = np.linalg.svd(Yc, full_matrices=False)
                    Uall[i] = U[:, :r_core]
                    Sall[i] = S[:r_core]
                    Vall[i] = Vh.conj().T[:, :r_core]
                np.save(u_path, Uall)
                np.save(s_path, Sall)
                np.save(v_path, Vall)
            for i in range(tl.shape[0]):
                self.xs.append(td[i])
                self.ys.append(tl[i])
                self.Us.append(Uall[i])
                self.Ss.append(Sall[i])
                self.Vs.append(Vall[i])
        self.n = len(self.xs)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        X = torch.from_numpy(self.xs[idx].copy()).float()
        Y = torch.from_numpy(self.ys[idx].copy()).float()
        U = torch.from_numpy(self.Us[idx]).to(torch.cfloat)
        S = torch.from_numpy(self.Ss[idx]).float()
        V = torch.from_numpy(self.Vs[idx]).to(torch.cfloat)
        return X, Y, U, S, V

def collate_fn(batch):
    X, Y, U, S, V = zip(*batch)
    X = torch.stack(X)
    Y = torch.stack(Y)
    U = torch.stack(U)
    S = torch.stack(S)
    V = torch.stack(V)
    Hin = to_complex(X)
    scale_mat = Hin.norm(dim=(-2, -1), keepdim=True).unsqueeze(-1).clamp_min(1e-8)
    scale_s = Hin.norm(dim=(-2, -1)).clamp_min(1e-8).reshape(-1, 1)
    return X/scale_mat, Y/scale_mat, U, S/scale_s, V

class CombinedModel(nn.Module):
    def __init__(self, uvnet: nn.Module = None, sref: nn.Module = None,
                 M: int = None, N: int = None, r: int = None, width: int = None,
                 ckpt_path: str = "best.pth"):
        super().__init__()
        if (uvnet is not None) and (sref is not None):
            self.uvnet = uvnet
            self.sref  = sref
            return
        if (M is None) or (N is None) or (r is None) or (width is None):
            if os.path.exists(ckpt_path):
                blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                if isinstance(blob, dict) and "state_dict" in blob and "arch" in blob:
                    arch = blob["arch"]
                    M = arch["M"]; N = arch["N"]; r = arch["r"]; width = arch["width"]
                else:
                    sd = blob if isinstance(blob, dict) else {}
                    def g(k1, k2):
                        return sd[k1] if k1 in sd else sd[k2]
                    w = g("uvnet.enc_x.0.weight", "enc_x.0.weight")
                    width = w.shape[0]
                    r_w = g("sref.net.2.weight", "net.2.weight")
                    r = r_w.shape[0]
                    u = g("uvnet.u_head.weight", "u_head.weight")
                    v = g("uvnet.v_head.weight", "v_head.weight")
                    M = u.shape[0] // (2 * r)
                    N = v.shape[0] // (2 * r)

        self.uvnet = UVNetFFT(M, N, r, width=width)
        self.sref  = SRefiner(r, hidden=width)

    def forward(self, x: Tensor):
        H = to_complex(x)
        U0, V0 = self.uvnet(x)
        U1, V1 = phase_align(U0, H, V0)
        s0, G = rayleigh_s(H, U1, V1)
        s1 = self.sref(G, s0)
        return s1

def get_avg_flops(model: nn.Module, input_data: Tensor) -> float:
    if input_data.dim() == 0 or input_data.size(0) == 0:
        raise RuntimeError("Input data must have a non-zero batch dimension")
    batch_size = input_data.size(0)
    model = model.eval().cpu()
    input_data = input_data.cpu()
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], with_flops=True, record_shapes=False) as prof:
            model(input_data)
    total_flops = sum(event.flops for event in prof.events())
    avg_flops = total_flops / batch_size
    return avg_flops * 1e-6 / 2

def read_cfg(cfg_path):
    lines = [l.strip() for l in open(cfg_path, 'r', encoding='utf-8') if l.strip()]
    M = int(''.join(filter(str.isdigit, lines[1])))
    N = int(''.join(filter(str.isdigit, lines[2])))
    r_full = int(''.join(filter(str.isdigit, lines[4])))
    return M, N, r_full

def infer_core(x, model, sref):
    Hc = to_complex(x)
    U0, V0 = model(x)
    U1, V1 = phase_align(U0, Hc, V0)
    s0, G = rayleigh_s(Hc, U1, V1)
    s1 = sref(G, s0)
    return s1, U1, V1

def save_npz(model, sref, test_path, out_path, device, C):
    X = np.load(test_path, mmap_mode='r')
    bs = 256
    U_list, S_list, V_list = [], [], []
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            xb = torch.from_numpy(X[i:i+bs].copy()).float().to(device)
            Hc = to_complex(xb)
            scale = Hc.norm(dim=(-2, -1), keepdim=True).unsqueeze(-1).clamp_min(1e-8)
            xb_n = xb / scale
            s1, Uc, Vc = infer_core(xb_n, model, sref)
            s_un = s1 * scale.reshape(-1, 1)
            U_list.append(from_complex(Uc).cpu().numpy())
            S_list.append(s_un.cpu().numpy())
            V_list.append(from_complex(Vc).cpu().numpy())
    U = np.concatenate(U_list, axis=0)
    S = np.concatenate(S_list, axis=0)
    V = np.concatenate(V_list, axis=0)
    np.savez(out_path, U=U, S=S, V=V, C=float(C))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="../../CompetitionData")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-2)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--lambda_off_start", type=float, default=0.1)
    p.add_argument("--lambda_off_end", type=float, default=0.1)
    p.add_argument("--lambda_mono", type=float, default=1e-3)
    p.add_argument("--lambda_sval", type=float, default=1e-2)
    p.add_argument("--r_core", type=int, default=32)
    p.add_argument("--model_pth", type=str, default="best.pth")
    p.add_argument("--export_dir", type=str, default="npz_export")
    p.add_argument("--data_pct", type=float, default=100.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = SVDDataset(args.data_dir, args.r_core)
    n_total = len(ds)
    n_subset = max(1, int(n_total * args.data_pct / 100.0))
    perm = torch.randperm(n_total)
    subset_idx = perm[:n_subset]
    n_val = max(1, min(n_subset - 1, int(0.10 * n_subset)))
    val_idx = subset_idx[:n_val]
    tr_idx = subset_idx[n_val:]

    class Subset(Dataset):
        def __init__(self, base, idxs):
            self.base, self.idxs = base, idxs
        def __len__(self):
            return len(self.idxs)
        def __getitem__(self, i):
            return self.base[self.idxs[i].item()]

    tr_loader = DataLoader(Subset(ds, tr_idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    va_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    M, N = ds.xs[0].shape[:2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UVNetFFT(M, N, args.r_core, width=args.width).to(device)
    sref = SRefiner(args.r_core, hidden=args.width).to(device)

    opt = torch.optim.AdamW(list(model.parameters()) + list(sref.parameters()), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    combined = CombinedModel(model, sref)
    dummy = torch.randn(1, M, N, 2)
    start_mmacs = get_avg_flops(combined, dummy)
    print(f"[{time.strftime('%H:%M:%S')}] MMACs at start: {start_mmacs:.2f}")
    model.to(device)
    sref.to(device)

    best = 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        sref.train()
        total_batches = len(tr_loader)
        acc_rec = acc_ort = acc_off = acc_sval = 0.0
        cnt = 0
        if args.epochs > 1:
            lam_off = args.lambda_off_start + (args.lambda_off_end - args.lambda_off_start) * (ep - 1) / (args.epochs - 1)
        else:
            lam_off = args.lambda_off_start
        for bi, (X, Y, _, St, _) in enumerate(tr_loader, 1):
            X, Y = X.to(device), Y.to(device)
            Hn = to_complex(X)
            U0, V0 = model(X)
            U1, V1 = phase_align(U0, Hn, V0)
            s0, G = rayleigh_s(Hn, U1, V1)
            s1 = sref(G, s0)
            s1, U1, V1 = sort_by_desc(s1, U1, V1)
            L_ae, rec, ort, off = loss_ae(Y, Hn, U1, s1, V1, lam_off)
            mono = mono_penalty(s1)
            L_s = F.smooth_l1_loss(torch.log(s1 + 1e-8), torch.log(St.to(device) + 1e-8))
            L = L_ae + args.lambda_mono * mono + args.lambda_sval * L_s
            opt.zero_grad()
            L.backward()
            opt.step()
            bs = X.size(0)
            acc_rec += rec.item() * bs
            acc_ort += ort.item() * bs
            acc_off += off.item() * bs
            acc_sval += L_s.item() * bs
            cnt += bs
            if bi % args.log_interval == 0 or bi == total_batches:
                print(f"[{time.strftime('%H:%M:%S')}] ep{ep:02d} {bi}/{total_batches} rec={acc_rec/cnt:.4f} ort={acc_ort/cnt:.4f} off={acc_off/cnt:.4f} sval={acc_sval/cnt:.4f} mono={mono.item():.4f} L={L.item():.4f}")
        sch.step()
        model.eval()
        sref.eval()
        vsum = osum = cnt = 0.0
        with torch.no_grad():
            for X, Y, _, _, _ in va_loader:
                X, Y = X.to(device), Y.to(device)
                Hn = to_complex(X)
                U0, V0 = model(X)
                U1, V1 = phase_align(U0, Hn, V0)
                s0, G = rayleigh_s(Hn, U1, V1)
                s1 = sref(G, s0)
                s1, U1, V1 = sort_by_desc(s1, U1, V1)
                _, rec, ort, off = loss_ae(Y, Hn, U1, s1, V1, lam_off)
                bs = X.size(0)
                vsum += rec.item() * bs
                osum += ort.item() * bs
                cnt += bs
            ae = (vsum + osum) / cnt
            print(f"[{time.strftime('%H:%M:%S')}] VAL ep{ep:02d} AE={ae:.6f}")
            if ae < best:
                best = ae
                torch.save(CombinedModel(model, sref).state_dict(), args.model_pth)

    os.makedirs(args.export_dir, exist_ok=True)

    combined = CombinedModel(model, sref)
    sd = torch.load(args.model_pth, map_location=device, weights_only=True)
    combined.load_state_dict(sd, strict=True)

    combined.eval()
    C = get_avg_flops(combined, dummy)
    combined.to(device)

    for k in (1, 2, 3):
        cfgp = os.path.join(args.data_dir, f"Round1CfgData{k}.txt")
        testp = os.path.join(args.data_dir, f"Round1TestData{k}.npy")
        outp = os.path.join(args.export_dir, f"{k}.npz")
        _, _, r_full = read_cfg(cfgp)
        assert r_full == args.r_core
        save_npz(model, sref, testp, outp, device, C)
        print(f"[{time.strftime('%H:%M:%S')}] Saved {outp}")

if __name__ == "__main__":
    main()