import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
	from PIL import Image
	import matplotlib.pyplot as plt
except Exception:
	Image = None  # type: ignore
	plt = None  # type: ignore

from drop_vis import compute_world_bounds_ul_quantile, world_to_image, TARGET_UNITS

TICK_SECONDS = 0.1


def load_al(al_path: Path) -> pd.DataFrame:
	df = pd.read_csv(al_path, dtype="object")
	# normalize types
	for col in ("armyindex", "rating", "human"):
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")
	return df


def print_drops_for_replay(root: Path, uid: int, drop_csv: Path) -> pd.DataFrame:
	df = pd.read_csv(drop_csv)
	df = df[df["replay_uid"] == uid].copy()
	if df.empty:
		print(f"No drops for replay uid={uid}")
		return df
	# Load players to map army->nickname
	al = root / str(uid) / "AL04.csv"
	name_map = {}
	try:
		al_df = load_al(al)
		for _, r in al_df.iterrows():
			if r.get("nickname") is not None and r.get("armyindex") == r.get("armyindex"):
				name_map[int(r["armyindex"])]= str(r["nickname"])  # type: ignore
	except Exception:
		pass
	# Print
	print(f"Drops in replay {uid}:")
	for _, r in df.sort_values("drop_tick").iterrows():
		army = int(r["army"])  # type: ignore
		name = name_map.get(army, f"Army {army}")
		mm = int(r["drop_time_s"] // 60)
		ss = int(r["drop_time_s"] % 60)
		unit = TARGET_UNITS.get(str(r["unit_id"]), str(r["unit_id"]))
		print(f"  {mm:02d}:{ss:02d}  {name:<20}  {unit:<9}  at ({r['drop_x']:.1f}, {r['drop_z']:.1f})")
	return df


def plot_replay_drops(root: Path, uid: int, drop_df: pd.DataFrame, map_image: Path, out_path: Path) -> None:
	if Image is None or plt is None:
		raise RuntimeError("Pillow/matplotlib not available")
	img = Image.open(map_image)
	w, h = img.size
	# Bounds from ALL units
	bounds = compute_world_bounds_ul_quantile(root, 0.01, 0.99)
	print(f"Bounds used: {bounds}")

	fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
	ax.imshow(img)
	ax.set_axis_off()
	for _, r in drop_df.iterrows():
		# fixed orientation: swap_axes=True, flip_x=True, flip_y=False
		px1, py1 = world_to_image(float(r["pickup_x"]), float(r["pickup_z"]), bounds, w, h, True, True, False)
		px2, py2 = world_to_image(float(r["drop_x"]), float(r["drop_z"]), bounds, w, h, True, True, False)
		px1 = float(np.clip(px1, 0, w - 1))
		py1 = float(np.clip(py1, 0, h - 1))
		px2 = float(np.clip(px2, 0, w - 1))
		py2 = float(np.clip(py2, 0, h - 1))
		unit_id = str(r.get("unit_id", ""))
		color = "#d32f2f" if unit_id == "xrl0305" else ("#0b1a44" if unit_id == "xel0305" else "#6a6a6a")
		ax.annotate("", xy=(px2, py2), xytext=(px1, py1), arrowprops=dict(arrowstyle="->", color=color, alpha=0.20, lw=1.8))
	plt.tight_layout(pad=0)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
	plt.close(fig)


def main() -> int:
	parser = argparse.ArgumentParser(description="Inspect drops for a single replay")
	parser.add_argument("--root", type=Path, default=Path("replays-analysis"))
	parser.add_argument("--uid", type=int, required=True)
	parser.add_argument("--drop-csv", type=Path, default=Path("analysis/outputs/drop_events.csv"))
	parser.add_argument("--map-image", type=Path, required=True)
	parser.add_argument("--out", type=Path, default=Path("analysis/outputs/replay_drops.png"))
	args = parser.parse_args()

	df = print_drops_for_replay(args.root, args.uid, args.drop_csv)
	if not df.empty:
		plot_replay_drops(args.root, args.uid, df, args.map_image, args.out)
		print(f"Saved: {args.out}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main()) 