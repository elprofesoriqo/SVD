## AI-Enabled Robust SVD Operator (Tech Arena 2025)

Neural-network approach to approximate the SVD operator for wireless MIMO channels under non-ideal conditions (noise, timing offsets). The model predicts top‑r singular values and corresponding unit‑norm left/right singular vectors without using traditional SVD/EVD inside the network. Scoring follows the competition metric: AE + model complexity (Mega MACs).

### Team
- @dec0dedd
- @kjedrasz2137
- @elprofesoriqo

### Repository structure
- `firstStage.py`: Round 1 training/inference and export to `.npz` (predicts `U, S, V`).
- `finalStage.py`: Round 2 training/inference and export (compact variant; uses `U = V` with phase alignment).
- `Task Description/`: Official background, task description, and helper `model_profiler.py` reference.

### Data
Download the official datasets and place files as provided by the organizers under:
- `./CompetitionData/` for Round 1 (`Round1*.npy`, `Round1CfgData*.txt`)
- `./CompetitionData2/` for Round 2 (`Round2*.npy`, `Round2CfgData*.txt`)

### License
Provided for the Tech Arena 2025 competition context. See task documents for usage constraints.


