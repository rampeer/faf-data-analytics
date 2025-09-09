# FAF Replay Analytics: Data‑Driven Insights

NB: Attend, adepts. The machine‑spirits were bound with sacred seals, soothed by incense, and compelled to reveal their secrets beneath the cog‑sigil of a vigilant Tech‑Priest. Binary hymns were sung. Errors were cast out. Truth was distilled. All cogitations and code herein have been reviewed and formally sanctioned by an experienced Tech‑Priest.

Within this sanctum we dissect Supreme Commander: Forged Alliance Forever (FAF) replays to wrench signal from the void: hard numbers, sober counsel, and curious surprises fit for presentation to the Balance Conclave. Our current rites hunt and expose transport‑borne drops of T3 land engines—most notably the Brick (`xrl0305`) and the Percival (`xel0305`)—projecting their paths upon the holy minimap and tabulating their deeds.

## End‑to‑end pipeline (quick start)
1) Detect drops (defaults include hardened filters: speed floor, dwell, displacement):
```bash
python analysis/drop_detection.py --root replays-analysis --quantile 0.90 --out-dir analysis/outputs
```
2) Visualize arrows + rating histograms, and print the uid with most drops:
```bash
python analysis/drop_vis.py --map-image analysis/setons.jpg --alpha 0.20 --bounds 0,1024,0,1024
```
3) (Optional) Inspect a specific replay to validate drops and render its own overlay:
```bash
python analysis/replay_inspect.py --uid 22646201 --map-image analysis/setons.jpg --drop-csv analysis/outputs/drop_events.csv
```

## Data layout

See `replays-analysis/DATA_DESCRIPTION.md` for detailed field descriptions.

Place replay exports under `replays-analysis/<uid>/`:
```
replays-analysis/
  <uid>/
    AL04.csv    # Players/armies
    EL04.csv    # Economy per army over time
    UL04.csv    # Units snapshots (positions, health, etc.)
    GL04.json   # Game metadata
```
Provide a minimap image (e.g., `analysis/setons.jpg`). All outputs go to `analysis/outputs/`.


## Setup
- Python 3.10+
- `pandas`, `numpy`, `matplotlib`, `pillow`

Install:
```bash
python -m pip install --user pandas numpy matplotlib pillow
```


## Usage

### 1) Detect drop events
Identify transport‑like relocations via per‑segment speed thresholds.
```bash
python analysis/drop_detection.py \
  --root replays-analysis \
  --quantile 0.90 \
  --min-duration-s 2.0 \
  --min-displacement 200 \
  --out-dir analysis/outputs
```
Defaults also enforce: `--min-transport-speed 3.6`, `--post-dwell-s 3`, `--post-dwell-speed 3.0` (tune if needed).

### 2) Visualize drops and ratings
Overlay semi‑transparent arrows on the minimap (Brick=red, Percival=dark navy). Plot rating histograms and print the replay with the most drops.
```bash
python analysis/drop_vis.py \
  --map-image analysis/setons.jpg \
  --alpha 0.20 \
  --bounds 0,1024,0,1024   # optional manual bounds
```
By default, bounds are derived from ALL units in `UL04.csv` (robust 1%–99% quantiles). Orientation is pre‑set for Seton’s Clutch.

Outputs: `drop_arrows.png`, `rating_hist.png`, `drop_stats.csv`.

### 3) Inspect a single replay
List every detected drop (time, owner, unit, coordinates) and render a per‑replay overlay.
```bash
python analysis/replay_inspect.py \
  --uid 22646201 \
  --map-image analysis/setons.jpg \
  --drop-csv analysis/outputs/drop_events.csv
```
Output: `replay_drops.png` and a console table with times (mm:ss), owners, and `(x, z)` coordinates.

## Roadmap and questions (Seton’s Clutch)
For now, the analysis focuses on Seton’s Clutch.
- T2→T3 conversion: who brings early T3 land power online faster and with fewer stalls?
- Late‑game roles: sustained value of heavy assault bots versus experimentals and static defenses.
- Drop effectiveness: how often do drops result in immediate, meaningful mass damage (e.g., mex snipes), and at which rating bands?
- Rating effects: do the same approaches succeed differently across rating tiers?
- Economy coupling: do energy/mass stalls correlate with missed timings or failed drops?

Contributions and new questions are welcome—add replays under `replays-analysis/<uid>/` and rerun the scripts. 