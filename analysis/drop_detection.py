import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
	import matplotlib.pyplot as plt
except Exception:  # matplotlib optional; we degrade gracefully
	plt = None


TICK_SECONDS = 0.1  # 10 ticks per second
TARGET_UNITS = {"xrl0305": "Brick", "xel0305": "Percival"}


@dataclass
class Thresholds:
	by_unit_type: Dict[str, float]
	quantile: float


@dataclass
class DropEvent:
	replay_uid: int
	army: int
	entity_id: int
	unit_id: str
	unit_label: str
	pickup_tick: int
	pickup_time_s: float
	pickup_x: float
	pickup_y: float
	pickup_z: float
	drop_tick: int
	drop_time_s: float
	drop_x: float
	drop_y: float
	drop_z: float
	duration_s: float
	displacement: float
	mean_speed: float
	max_speed: float
	segment_points: int
	threshold_used: float


def find_ul_files(root: Path) -> List[Path]:
	return sorted(root.glob("**/UL04.csv"))


def compute_speeds_for_file(ul_path: Path, unit_ids: Iterable[str]) -> Dict[str, np.ndarray]:
	"""Return per-unit-type speed samples (magnitude) for threshold estimation.
	Speeds are computed on XZ plane between consecutive samples per EntityId.
	"""
	usecols = [
		"GameTick",
		"Army",
		"EntityId",
		"UnitId",
		"FractionComplete",
		"Position.x",
		"Position.y",
		"Position.z",
	]
	dtypes = {
		"GameTick": "int32",
		"Army": "int16",
		"EntityId": "int64",
		"UnitId": "category",
		"FractionComplete": "float32",
		"Position.x": "float32",
		"Position.y": "float32",
		"Position.z": "float32",
	}
	try:
		df = pd.read_csv(ul_path, usecols=usecols, dtype=dtypes)
	except Exception as exc:
		print(f"Failed to read {ul_path}: {exc}", file=sys.stderr)
		return {u: np.array([], dtype=np.float32) for u in unit_ids}

	# Filter only target units and completed builds
	df = df[df["UnitId"].isin(list(unit_ids)) & (df["FractionComplete"] >= 1.0)].copy()
	if df.empty:
		return {u: np.array([], dtype=np.float32) for u in unit_ids}

	df.sort_values(["EntityId", "GameTick"], inplace=True, kind="mergesort")
	# Compute per-entity diffs
	for col in ("Position.x", "Position.z", "GameTick"):
		df[f"_diff_{col}"] = df.groupby("EntityId")[col].diff()

	# delta time in seconds between consecutive samples
	df["_dt_s"] = df["_diff_GameTick"] * TICK_SECONDS
	# distance in XZ plane
	df["_dx"] = df["_diff_Position.x"]
	df["_dz"] = df["_diff_Position.z"]
	df["_dist"] = np.sqrt(df["_dx"] * df["_dx"] + df["_dz"] * df["_dz"])
	# speed = distance / time, guard against zero/neg dt
	df["_speed"] = df["_dist"] / df["_dt_s"].replace({0.0: np.nan})

	result: Dict[str, np.ndarray] = {}
	for u in unit_ids:
		speeds = df.loc[(df["UnitId"] == u) & (df["_dt_s"] > 0), "_speed"].to_numpy(dtype=np.float32)
		# Drop NaN/inf
		speeds = speeds[np.isfinite(speeds)]
		result[u] = speeds
	return result


def estimate_thresholds(ul_files: List[Path], quantile: float, unit_ids: Iterable[str]) -> Thresholds:
	by_unit: Dict[str, List[np.ndarray]] = {u: [] for u in unit_ids}
	for ul in ul_files:
		per_file = compute_speeds_for_file(ul, unit_ids)
		for u in unit_ids:
			if per_file[u].size:
				by_unit[u].append(per_file[u])

	by_unit_threshold: Dict[str, float] = {}
	for u in unit_ids:
		if len(by_unit[u]) == 0:
			by_unit_threshold[u] = float("nan")
			continue
		all_speeds = np.concatenate(by_unit[u])
		# Remove extremely small jitters
		all_speeds = all_speeds[all_speeds > 1e-3]
		if all_speeds.size == 0:
			by_unit_threshold[u] = float("nan")
			continue
		thr = float(np.quantile(all_speeds, quantile))
		by_unit_threshold[u] = thr
	return Thresholds(by_unit_threshold, quantile)


def replay_uid_from_path(ul_path: Path) -> Optional[int]:
	try:
		return int(ul_path.parent.name)
	except Exception:
		return None


def detect_drop_events_in_file(
	ul_path: Path,
	thresholds: Thresholds,
	min_duration_s: float,
	min_displacement: float,
	max_tick_gap: Optional[int],
	unit_ids: Iterable[str],
	min_transport_speed: float,
	post_dwell_s: float,
	post_dwell_speed: float,
) -> List[DropEvent]:
	usecols = [
		"GameTick",
		"Army",
		"EntityId",
		"UnitId",
		"FractionComplete",
		"Position.x",
		"Position.y",
		"Position.z",
	]
	dtypes = {
		"GameTick": "int32",
		"Army": "int16",
		"EntityId": "int64",
		"UnitId": "category",
		"FractionComplete": "float32",
		"Position.x": "float32",
		"Position.y": "float32",
		"Position.z": "float32",
	}
	try:
		df = pd.read_csv(ul_path, usecols=usecols, dtype=dtypes)
	except Exception as exc:
		print(f"Failed to read {ul_path}: {exc}", file=sys.stderr)
		return []

	df = df[df["UnitId"].isin(list(unit_ids)) & (df["FractionComplete"] >= 1.0)].copy()
	if df.empty:
		return []

	df.sort_values(["EntityId", "GameTick"], inplace=True, kind="mergesort")
	# Compute diffs
	for col in ("Position.x", "Position.y", "Position.z", "GameTick"):
		df[f"_diff_{col}"] = df.groupby("EntityId")[col].diff()
	# Derived
	df["_dt_s"] = df["_diff_GameTick"] * TICK_SECONDS
	df["_dx"] = df["_diff_Position.x"]
	df["_dz"] = df["_diff_Position.z"]
	df["_dist"] = np.sqrt(df["_dx"] * df["_dx"] + df["_dz"] * df["_dz"])
	df["_speed"] = df["_dist"] / df["_dt_s"].replace({0.0: np.nan})

	# Optional: filter out massive tick gaps if requested
	if max_tick_gap is not None:
		df = df[(df["_diff_GameTick"].isna()) | (df["_diff_GameTick"] <= max_tick_gap)]

	replay_uid = replay_uid_from_path(ul_path)
	results: List[DropEvent] = []

	# Iterate per entity to find high-speed segments
	for (unit_id, entity_id), g in df.groupby(["UnitId", "EntityId"], sort=False):
		g = g.reset_index(drop=True)
		if g.shape[0] < 2:
			continue
		thr = thresholds.by_unit_type.get(str(unit_id), float("nan"))
		if not math.isfinite(thr):
			continue
		# Boolean series marks segments ending at each row
		# High-speed only if both above quantile threshold and absolute floor
		is_high = (g["_dt_s"] > 0) & (g["_speed"] >= max(thr, min_transport_speed))
		# Find contiguous regions where is_high is True
		# We consider transitions in is_high to define segment boundaries
		prev = False
		start_idx: Optional[int] = None
		for i, high in enumerate(is_high):
			if high and not prev:
				start_idx = i
			elif (not high) and prev:
				# Segment ended at i-1
				end_idx = i - 1
				# Determine pickup (preceding sample) and drop (following sample) indices
				pickup_idx = max(0, start_idx - 1)
				drop_idx = min(g.shape[0] - 1, end_idx + 1)
				if pickup_idx == drop_idx:
					start_idx = None
					prev = False
					continue
				# Compute metrics across the high-speed span [start_idx..end_idx]
				span = g.loc[start_idx : end_idx]
				duration_s = float(span["_dt_s"].fillna(0.0).sum())
				if duration_s < min_duration_s:
					start_idx = None
					prev = False
					continue
				# Displacement between pickup and drop positions (XZ)
				px, py, pz = (
					float(g.loc[pickup_idx, "Position.x"]),
					float(g.loc[pickup_idx, "Position.y"]),
					float(g.loc[pickup_idx, "Position.z"]),
				)
				dx, dz = (
					float(g.loc[drop_idx, "Position.x"]) - px,
					float(g.loc[drop_idx, "Position.z"]) - pz,
				)
				displacement = math.hypot(dx, dz)
				if displacement < min_displacement:
					start_idx = None
					prev = False
					continue
				# Post-drop dwell: ensure unit slows down after drop_idx
				dwell_ok = True
				if post_dwell_s > 0:
					# accumulate dt from drop_idx forward until reaching post_dwell_s threshold
					acc_s = 0.0
					j = drop_idx + 1
					while j < g.shape[0] and acc_s < post_dwell_s:
						seg_dt = float(g.loc[j, "_dt_s"]) if g.loc[j, "_dt_s"] == g.loc[j, "_dt_s"] else 0.0
						seg_speed = float(g.loc[j, "_speed"]) if g.loc[j, "_speed"] == g.loc[j, "_speed"] else 0.0
						# If during dwell window speed exceeds allowed, reject
						if seg_speed > post_dwell_speed:
							dwell_ok = False
							break
						acc_s += seg_dt
						j += 1
				if not dwell_ok:
					start_idx = None
					prev = False
					continue
				mean_speed = float(span["_speed"].replace([np.inf, -np.inf], np.nan).dropna().mean())
				max_speed = float(span["_speed"].replace([np.inf, -np.inf], np.nan).dropna().max())
				army = int(g.loc[drop_idx, "Army"]) if not pd.isna(g.loc[drop_idx, "Army"]) else int(g.loc[pickup_idx, "Army"])  # type: ignore
				pickup_tick = int(g.loc[pickup_idx, "GameTick"]) if g.loc[pickup_idx, "GameTick"] == g.loc[pickup_idx, "GameTick"] else int(g.loc[start_idx, "GameTick"])  # NaN-safe
				drop_tick = int(g.loc[drop_idx, "GameTick"]) if g.loc[drop_idx, "GameTick"] == g.loc[drop_idx, "GameTick"] else int(g.loc[end_idx, "GameTick"])  # NaN-safe
				results.append(
					DropEvent(
						replay_uid=replay_uid or -1,
						army=army,
						entity_id=int(entity_id),
						unit_id=str(unit_id),
						unit_label=TARGET_UNITS.get(str(unit_id), str(unit_id)),
						pickup_tick=pickup_tick,
						pickup_time_s=pickup_tick * TICK_SECONDS,
						pickup_x=px,
						pickup_y=py,
						pickup_z=pz,
						drop_tick=drop_tick,
						drop_time_s=drop_tick * TICK_SECONDS,
						drop_x=float(g.loc[drop_idx, "Position.x"]),
						drop_y=float(g.loc[drop_idx, "Position.y"]),
						drop_z=float(g.loc[drop_idx, "Position.z"]),
						duration_s=duration_s,
						displacement=displacement,
						mean_speed=mean_speed,
						max_speed=max_speed,
						segment_points=int(span.shape[0]),
						threshold_used=thr,
					)
				)
				start_idx = None
			prev = bool(high)
		# If ended high, we ignore incomplete trailing segment

	return results


def save_histograms(thresholds: Thresholds, ul_files: List[Path], unit_ids: Iterable[str], out_dir: Path, bins: int) -> None:
	if plt is None:
		print("matplotlib not available; skipping histogram plots", file=sys.stderr)
		return

	# Aggregate speeds again for plotting (kept simple; could be cached)
	agg: Dict[str, List[np.ndarray]] = {u: [] for u in unit_ids}
	for ul in ul_files:
		per_file = compute_speeds_for_file(ul, unit_ids)
		for u in unit_ids:
			if per_file[u].size:
				agg[u].append(per_file[u])

	out_dir.mkdir(parents=True, exist_ok=True)
	for u in unit_ids:
		if len(agg[u]) == 0:
			continue
		data = np.concatenate(agg[u])
		data = data[np.isfinite(data) & (data > 0)]
		if data.size == 0:
			continue
		thr = thresholds.by_unit_type.get(u, float("nan"))
		plt.figure(figsize=(8, 4))
		plt.hist(data, bins=bins, color="#4472c4", alpha=0.8)
		if math.isfinite(thr):
			plt.axvline(thr, color="#c00000", linestyle="--", label=f"q={thresholds.quantile:.3f}: {thr:.2f}")
		plt.title(f"Speed histogram for {TARGET_UNITS.get(u, u)} ({u})")
		plt.xlabel("Speed (map units / s)")
		plt.ylabel("Count")
		if math.isfinite(thr):
			plt.legend()
		plt.tight_layout()
		png_path = out_dir / f"speed_hist_{u}.png"
		plt.savefig(png_path, dpi=150)
		plt.close()


def write_events_csv(events: List[DropEvent], out_csv: Path) -> None:
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	rows = [
		{
			"replay_uid": e.replay_uid,
			"army": e.army,
			"entity_id": e.entity_id,
			"unit_id": e.unit_id,
			"unit_label": e.unit_label,
			"pickup_tick": e.pickup_tick,
			"pickup_time_s": e.pickup_time_s,
			"pickup_x": e.pickup_x,
			"pickup_y": e.pickup_y,
			"pickup_z": e.pickup_z,
			"drop_tick": e.drop_tick,
			"drop_time_s": e.drop_time_s,
			"drop_x": e.drop_x,
			"drop_y": e.drop_y,
			"drop_z": e.drop_z,
			"duration_s": e.duration_s,
			"displacement": e.displacement,
			"mean_speed": e.mean_speed,
			"max_speed": e.max_speed,
			"segment_points": e.segment_points,
			"threshold_used": e.threshold_used,
		}
		for e in events
	]
	pd.DataFrame(rows).to_csv(out_csv, index=False)


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Detect transport-like drops for Bricks and Percivals using speed thresholds")
	parser.add_argument("--root", type=Path, default=Path("replays-analysis"), help="Root folder containing replay subdirectories")
	parser.add_argument("--quantile", type=float, default=0.9, help="Quantile to set high-speed threshold per unit type")
	parser.add_argument("--min-transport-speed", type=float, default=3.6, help="Absolute speed floor to consider transport (map units/s); must exceed ground max (~2.95)")
	parser.add_argument("--min-duration-s", type=float, default=2.0, help="Minimum duration of high-speed segment to count as transport")
	parser.add_argument("--min-displacement", type=float, default=200.0, help="Minimum pickup->drop displacement (map units)")
	parser.add_argument("--max-tick-gap", type=int, default=1800, help="Ignore segments with delta ticks greater than this (None to disable)")
	parser.add_argument("--post-dwell-s", type=float, default=3.0, help="Require at least this many seconds of low speed after drop to confirm unload")
	parser.add_argument("--post-dwell-speed", type=float, default=3.0, help="Max speed during dwell (map units/s); set ~1.0–1.5× ground speed")
	parser.add_argument("--hist-bins", type=int, default=120, help="Number of bins for histograms")
	parser.add_argument("--out-dir", type=Path, default=Path("analysis/outputs"), help="Output directory for artifacts")
	args = parser.parse_args(argv)

	ul_files = find_ul_files(args.root)
	if not ul_files:
		print(f"No UL04.csv files found under {args.root}", file=sys.stderr)
		return 2

	unit_ids = list(TARGET_UNITS.keys())
	# Pass 1: thresholds
	thresholds = estimate_thresholds(ul_files, args.quantile, unit_ids)
	# Optional plots
	save_histograms(thresholds, ul_files, unit_ids, args.out_dir, args.hist_bins)

	# Pass 2: events
	all_events: List[DropEvent] = []
	for ul in ul_files:
		file_events = detect_drop_events_in_file(
			ul,
			thresholds,
			min_duration_s=args.min_duration_s,
			min_displacement=args.min_displacement,
			max_tick_gap=None if args.max_tick_gap <= 0 else args.max_tick_gap,
			unit_ids=unit_ids,
			min_transport_speed=args.min_transport_speed,
			post_dwell_s=args.post_dwell_s,
			post_dwell_speed=args.post_dwell_speed,
		)
		if file_events:
			all_events.extend(file_events)

	out_csv = args.out_dir / "drop_events.csv"
	write_events_csv(all_events, out_csv)
	print(f"Wrote {len(all_events)} drop events to {out_csv}")
	return 0


if __name__ == "__main__":
	sys.exit(main()) 