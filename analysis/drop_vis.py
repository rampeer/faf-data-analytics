import argparse
import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
	from PIL import Image
	import matplotlib.pyplot as plt
except Exception:
	Image = None  # type: ignore
	plt = None  # type: ignore


TICK_SECONDS = 0.1
TARGET_UNITS = {"xrl0305": "Brick", "xel0305": "Percival"}


def find_replay_dirs(root: Path) -> List[Path]:
	return sorted([p for p in root.iterdir() if p.is_dir()])


def compute_world_bounds(root: Path) -> Tuple[float, float, float, float]:
	"""Scan UL04.csv files to find global min/max for X and Z.
	Returns (min_x, max_x, min_z, max_z).
	"""
	min_x = math.inf
	max_x = -math.inf
	min_z = math.inf
	max_z = -math.inf
	ul_files = sorted(root.glob("**/UL04.csv"))
	usecols = ["Position.x", "Position.z"]
	dtypes = {"Position.x": "float32", "Position.z": "float32"}
	for ul in ul_files:
		try:
			df = pd.read_csv(ul, usecols=usecols, dtype=dtypes)
		except Exception:
			continue
		if df.empty:
			continue
		cx = df["Position.x"].to_numpy()
		cz = df["Position.z"].to_numpy()
		if cx.size == 0:
			continue
		min_x = min(min_x, float(np.nanmin(cx)))
		max_x = max(max_x, float(np.nanmax(cx)))
		min_z = min(min_z, float(np.nanmin(cz)))
		max_z = max(max_z, float(np.nanmax(cz)))
	# Fallbacks if bounds are missing
	if not math.isfinite(min_x) or not math.isfinite(max_x) or not math.isfinite(min_z) or not math.isfinite(max_z):
		# Default square map 0..1024
		return (0.0, 1024.0, 0.0, 1024.0)
	return (min_x, max_x, min_z, max_z)


def compute_world_bounds_ul_quantile(root: Path, q_low: float, q_high: float) -> Tuple[float, float, float, float]:
	"""Quantile-based robust bounds from all UL04.csv data to ignore outliers (uses ALL units)."""
	xs: List[np.ndarray] = []
	zs: List[np.ndarray] = []
	ul_files = sorted(root.glob("**/UL04.csv"))
	usecols = ["Position.x", "Position.z"]
	dtypes = {"Position.x": "float32", "Position.z": "float32"}
	for ul in ul_files:
		try:
			df = pd.read_csv(ul, usecols=usecols, dtype=dtypes)
		except Exception:
			continue
		if df.empty:
			continue
		x = df["Position.x"].to_numpy(np.float32)
		z = df["Position.z"].to_numpy(np.float32)
		if x.size:
			xs.append(x[np.isfinite(x)])
		if z.size:
			zs.append(z[np.isfinite(z)])
	if not xs or not zs:
		return compute_world_bounds(root)
	all_x = np.concatenate(xs)
	all_z = np.concatenate(zs)
	min_x = float(np.quantile(all_x, q_low))
	max_x = float(np.quantile(all_x, q_high))
	min_z = float(np.quantile(all_z, q_low))
	max_z = float(np.quantile(all_z, q_high))
	return (min_x, max_x, min_z, max_z)


def compute_bounds_from_events(drop_csv: Path, margin: float = 0.02) -> Tuple[float, float, float, float]:
	"""Bounds from drop events themselves with a small margin (not preferred)."""
	df = pd.read_csv(drop_csv)
	if df.empty:
		return (0.0, 1024.0, 0.0, 1024.0)
	x_vals = pd.concat([df["pickup_x"], df["drop_x"]]).to_numpy()
	z_vals = pd.concat([df["pickup_z"], df["drop_z"]]).to_numpy()
	min_x, max_x = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
	min_z, max_z = float(np.nanmin(z_vals)), float(np.nanmax(z_vals))
	# Add margins
	dx = (max_x - min_x) or 1.0
	dz = (max_z - min_z) or 1.0
	min_x -= dx * margin
	max_x += dx * margin
	min_z -= dz * margin
	max_z += dz * margin
	return (min_x, max_x, min_z, max_z)


def world_to_image(
	x: float,
	z: float,
	bounds: Tuple[float, float, float, float],
	width: int,
	height: int,
	swap_axes: bool,
	flip_x: bool,
	flip_y: bool,
) -> Tuple[float, float]:
	"""Scale world XZ to image pixel coordinates with orientation controls.
	- swap_axes: if True, map world Z to image X, and world X to image Y.
	- flip_x / flip_y: mirror the respective image axes after swap.
	Image coordinates are top-left origin.
	"""
	min_x, max_x, min_z, max_z = bounds
	if max_x <= min_x or max_z <= min_z:
		return (0.0, 0.0)
	nx = (x - min_x) / (max_x - min_x)
	nz = (z - min_z) / (max_z - min_z)
	# Base normalized mapping without swap: px<-nx, py<-(1-nz)
	px_n = nx
	py_n = 1.0 - nz
	if swap_axes:
		px_n, py_n = py_n, px_n
	if flip_x:
		px_n = 1.0 - px_n
	if flip_y:
		py_n = 1.0 - py_n
	px = px_n * (width - 1)
	py = py_n * (height - 1)
	return (px, py)


def draw_drop_arrows(map_image_path: Path, drop_csv: Path, bounds: Tuple[float, float, float, float], out_path: Path, alpha: float = 0.20, swap_axes: bool = True, flip_x: bool = True, flip_y: bool = False) -> None:
	if Image is None or plt is None:
		raise RuntimeError("Pillow/matplotlib not available")
	img = Image.open(map_image_path)
	w, h = img.size

	# Diagnostics: print bounds and image size
	min_x, max_x, min_z, max_z = bounds
	print(f"Map image size: width={w}, height={h}")
	print(f"World bounds used: min_x={min_x:.2f}, max_x={max_x:.2f}, min_z={min_z:.2f}, max_z={max_z:.2f}")
	print(f"Orientation: swap_axes={swap_axes}, flip_x={flip_x}, flip_y={flip_y}")

	df = pd.read_csv(drop_csv)
	print(f"Drop events loaded: {len(df)}")
	if not df.empty and {"replay_uid"}.issubset(df.columns):
		grp = df.groupby("replay_uid").size().reset_index(name="drops_count").sort_values("drops_count", ascending=False)
		if not grp.empty:
			top_uid = int(grp.iloc[0]["replay_uid"])  # type: ignore
			top_cnt = int(grp.iloc[0]["drops_count"])  # type: ignore
			print(f"Top replay by drops: uid={top_uid} (drops={top_cnt})")
	if df.empty:
		img.save(out_path)
		return

	# Diagnostics for drop coordinates
	for col in ("pickup_x", "pickup_z", "drop_x", "drop_z"):
		if col in df.columns and len(df[col]):
			print(f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}")

	fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
	ax.imshow(img)
	ax.set_axis_off()

	drawn = 0
	for _, row in df.iterrows():
		ux1, uz1 = float(row["pickup_x"]), float(row["pickup_z"])  # world
		ux2, uz2 = float(row["drop_x"]), float(row["drop_z"])  # world
		px1, py1 = world_to_image(ux1, uz1, bounds, w, h, swap_axes=swap_axes, flip_x=flip_x, flip_y=flip_y)
		px2, py2 = world_to_image(ux2, uz2, bounds, w, h, swap_axes=swap_axes, flip_x=flip_x, flip_y=flip_y)
		# Clamp to image bounds
		px1 = float(np.clip(px1, 0, w - 1))
		py1 = float(np.clip(py1, 0, h - 1))
		px2 = float(np.clip(px2, 0, w - 1))
		py2 = float(np.clip(py2, 0, h - 1))

		unit_id = str(row.get("unit_id", ""))
		color = "#d32f2f" if unit_id == "xrl0305" else ("#0b1a44" if unit_id == "xel0305" else "#6a6a6a")
		ax.annotate(
			"",
			xy=(px2, py2),
			xytext=(px1, py1),
			arrowprops=dict(arrowstyle="->", color=color, alpha=alpha, lw=1.8),
		)
		drawn += 1

	print(f"Arrows drawn: {drawn}")
	plt.tight_layout(pad=0)
	fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
	plt.close(fig)


def rating_histograms(root: Path, drop_csv: Path, out_path: Path) -> None:
	if plt is None:
		raise RuntimeError("matplotlib not available")
	# Determine droppers set of (uid, army)
	drops = pd.read_csv(drop_csv)
	drop_pairs = set()
	if not drops.empty:
		for _, r in drops[["replay_uid", "army"]].dropna().iterrows():
			try:
				uid = int(r["replay_uid"])  # type: ignore
				army = int(r["army"])  # type: ignore
				drop_pairs.add((uid, army))
			except Exception:
				continue

	all_ratings: List[float] = []
	dropper_ratings: List[float] = []
	for d in find_replay_dirs(root):
		al = d / "AL04.csv"
		if not al.exists():
			continue
		try:
			df = pd.read_csv(al, dtype="object")
		except Exception:
			continue
		# Keep human players only
		try:
			df["human"] = pd.to_numeric(df["human"], errors="coerce")
			mask_human = df["human"] == 1
		except Exception:
			mask_human = np.ones(len(df), dtype=bool)
		try:
			df["armyindex"] = pd.to_numeric(df["armyindex"], errors="coerce")
		except Exception:
			pass
		try:
			df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
		except Exception:
			df["rating"] = np.nan
		mask_rating = df["rating"].notna()
		df = df[mask_human & mask_rating]
		if df.empty:
			continue
		uid = None
		try:
			uid = int(d.name)
		except Exception:
			uid = None
		for _, r in df.iterrows():
			rating = float(r["rating"])  # type: ignore
			all_ratings.append(rating)
			if uid is not None and (uid, int(r["armyindex"])) in drop_pairs:
				dropper_ratings.append(rating)

	# Plot
	fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
	bins = 30
	axes[0].hist(all_ratings, bins=bins, color="#444", alpha=0.8)
	axes[0].set_title("All players ratings")
	axes[0].set_xlabel("Rating")
	axes[0].set_ylabel("Count")
	axes[1].hist(dropper_ratings, bins=bins, color="#0b1a44", alpha=0.8)
	axes[1].set_title("Droppers ratings")
	axes[1].set_xlabel("Rating")
	axes[1].set_ylabel("Count")
	plt.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path)
	plt.close(fig)


def write_drop_stats(drop_csv: Path, out_path: Path) -> None:
	df = pd.read_csv(drop_csv)
	if df.empty:
		pd.DataFrame({"unit_label": [], "count": []}).to_csv(out_path, index=False)
		return
	# Prefer unit_label if present; else derive from unit_id
	if "unit_label" not in df.columns:
		df["unit_label"] = df["unit_id"].map(lambda u: TARGET_UNITS.get(str(u), str(u)))
	counts = df.groupby("unit_label").size().reset_index(name="count")
	# Ensure both categories present
	for label in ("Brick", "Percival"):
		if label not in set(counts["unit_label"].astype(str)):
			counts = pd.concat([counts, pd.DataFrame([[label, 0]], columns=["unit_label", "count"])], ignore_index=True)
	counts.sort_values("unit_label", inplace=True)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	counts.to_csv(out_path, index=False)


def main() -> int:
	parser = argparse.ArgumentParser(description="Visualize Brick/Percival drops and rating distributions")
	parser.add_argument("--root", type=Path, default=Path("replays-analysis"))
	parser.add_argument("--drop-csv", type=Path, default=Path("analysis/outputs/drop_events.csv"))
	parser.add_argument("--map-image", type=Path, required=True, help="Path to minimap image, e.g., setons.jpg")
	parser.add_argument("--out-dir", type=Path, default=Path("analysis/outputs"))
	parser.add_argument("--alpha", type=float, default=0.20)
	parser.add_argument("--bounds", type=str, default="", help="Manual world bounds as 'min_x,max_x,min_z,max_z' (overrides auto)")
	args = parser.parse_args()

	# Derive world bounds from ALL units using UL04.csv with robust quantiles
	q_low, q_high = 0.01, 0.99
	auto_bounds = compute_world_bounds_ul_quantile(args.root, q_low, q_high)
	bounds = auto_bounds
	bounds_source = f"UL quantiles q=[{q_low}, {q_high}] from ALL units"

	# Optional manual override
	if args.bounds:
		try:
			parts = [float(p) for p in args.bounds.split(",")]
			if len(parts) == 4:
				bounds = (parts[0], parts[1], parts[2], parts[3])
				bounds_source = "manual"
			else:
				print("Warning: --bounds must have 4 comma-separated numbers; ignoring manual bounds")
		except Exception:
			print("Warning: failed to parse --bounds; ignoring manual bounds")

	print(f"Bounds source: {bounds_source}")

	# Visual arrows (fixed orientation: swap_axes=True, flip_x=True, flip_y=False)
	draw_drop_arrows(
		args.map_image,
		args.drop_csv,
		bounds,
		args.out_dir / "drop_arrows.png",
		alpha=args.alpha,
	)
	# Rating histograms
	rating_histograms(args.root, args.drop_csv, args.out_dir / "rating_hist.png")
	# Stats
	write_drop_stats(args.drop_csv, args.out_dir / "drop_stats.csv")
	print("Saved: drop_arrows.png, rating_hist.png, drop_stats.csv")
	return 0


if __name__ == "__main__":
	raise SystemExit(main()) 