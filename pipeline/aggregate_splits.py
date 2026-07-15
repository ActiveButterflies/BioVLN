#!/usr/bin/env python3
"""
Aggregate per-scene episodes into train/val/test splits.

Reads all *_episodes.json.gz files from scene output directories,
groups by scene split assignment, merges, and validates.

Usage:
  python aggregate_splits.py --build_manifest output/build_manifest.json

  # Or scan a directory for scenes
  python aggregate_splits.py --scenes_dir output/ --output_dir splits/

  # Validate existing splits
  python aggregate_splits.py --validate splits/
"""

import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set


def _find_episodes(scenes_dir: Path) -> List[Dict]:
    """Scan scenes_dir for per-scene episode files and scene split assignments.

    First looks for a build_manifest.json, then falls back to scanning
    *_spec.json or reading split from episode metadata.
    """
    # Try build manifest first
    manifest_path = scenes_dir / "build_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

        episodes_info = []
        for entry in manifest:
            scene_name = entry["scene_name"]
            split = entry["split"]
            scene_out = Path(entry.get("output_dir", ""))
            if not scene_out.exists():
                scene_out = scenes_dir / scene_name

            ep_gz = scene_out / f"{scene_name}_episodes.json.gz"
            if ep_gz.exists():
                episodes_info.append({
                    "scene_name": scene_name,
                    "split": split,
                    "episodes_path": ep_gz,
                })
            else:
                print(f"  ⚠  No episodes found for {scene_name} at {ep_gz}")
        return episodes_info

    # Fallback: scan subdirectories
    episodes_info = []
    for subdir in sorted(scenes_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Check for spec file
        specs = list(subdir.glob("*_spec.json"))
        if specs:
            with open(specs[0]) as f:
                spec = json.load(f)
            split = spec.get("split", "train")
        else:
            split = "train"  # default

        scene_name = subdir.name
        ep_gz = subdir / f"{scene_name}_episodes.json.gz"
        if ep_gz.exists():
            episodes_info.append({
                "scene_name": scene_name,
                "split": split,
                "episodes_path": ep_gz,
            })

    return episodes_info


def aggregate(episodes_info: List[Dict],
              output_dir: Path,
              category_mapping: Dict[str, int] = None) -> Dict:
    """
    Merge per-scene episodes into train/val/test splits.

    Args:
        episodes_info: list of {scene_name, split, episodes_path}
        output_dir: where to write episodes_{split}.json.gz
        category_mapping: dict category_name → int id (auto-detected if None)

    Returns:
        stats dict
    """
    # Collect whatever split names appear in the data (dev, val, test, train, etc.)
    splits: Dict[str, List[dict]] = {}
    scene_splits: Dict[str, Set[str]] = {}  # split → set of scene names

    # Auto-detect category mapping from first episode file
    if category_mapping is None:
        category_mapping = _detect_category_mapping(episodes_info)

    for info in episodes_info:
        scene_name = info["scene_name"]
        split = info["split"]
        ep_path = info["episodes_path"]

        with gzip.open(ep_path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        episodes = data.get("episodes", [])

        # Tag each episode with its scene
        for ep in episodes:
            ep["scene_id"] = f"{scene_name}.glb"
            ep["scene_name"] = scene_name

        splits.setdefault(split, []).extend(episodes)
        scene_splits.setdefault(split, set()).add(scene_name)

    # ── Validate: no scene leakage across splits ──────────────────────
    split_names = sorted(splits.keys())
    for i, a in enumerate(split_names):
        for b in split_names[i+1:]:
            overlap = scene_splits.get(a, set()) & scene_splits.get(b, set())
            if overlap:
                print(f"⚠  SCENE LEAKAGE: {a} ∩ {b}: {overlap}")

    # ── Renumber episode IDs, write per-split files ────────────────────
    stats = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in split_names:
        episodes = splits[split_name]
        if not episodes:
            print(f"  ⚠  No episodes for split '{split_name}'")
            continue
        if not episodes:
            print(f"  ⚠  No episodes for split '{split_name}'")
            continue

        for i, ep in enumerate(episodes):
            ep["episode_id"] = str(i)

        output = {
            "episodes": episodes,
            "category_to_task_category_id": category_mapping,
            "category_to_scene_annotation_category_id": {},
        }

        out_path = output_dir / f"episodes_{split_name}.json.gz"
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"  ✓ {out_path} ({len(episodes)} episodes)")

        # Stats
        cat_counts = Counter(ep.get("object_category", "?") for ep in episodes)
        scene_counts = Counter(ep.get("scene_name", "?") for ep in episodes)
        geo_dists = [ep.get("info", {}).get("geodesic_distance", -1)
                     for ep in episodes]

        stats[split_name] = {
            "total": len(episodes),
            "scenes": len(scene_counts),
            "scene_names": sorted(scene_counts.keys()),
            "categories": dict(cat_counts),
            "geodesic_avg": (
                sum(d for d in geo_dists if d > 0) /
                max(1, sum(1 for d in geo_dists if d > 0))
            ),
            "geodesic_min": min(d for d in geo_dists if d > 0) if any(d > 0 for d in geo_dists) else -1,
            "geodesic_max": max(geo_dists),
        }

    return stats


def _detect_category_mapping(episodes_info: List[Dict]) -> Dict[str, int]:
    """Auto-detect category_to_task_category_id from first episode file."""
    for info in episodes_info:
        ep_path = info["episodes_path"]
        with gzip.open(ep_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        mapping = data.get("category_to_task_category_id")
        if mapping:
            return mapping
    return {}


def print_stats(stats: Dict):
    """Print split statistics."""
    print(f"\n{'='*60}")
    print("SPLIT STATISTICS")
    print(f"{'='*60}")

    total_all = 0
    for split_name in sorted(stats.keys()):
        s = stats.get(split_name, {})
        total = s.get("total", 0)
        total_all += total
        scenes = s.get("scenes", 0)
        print(f"\n[{split_name.upper()}] {total} episodes, {scenes} scenes")
        print(f"  Scenes: {', '.join(s.get('scene_names', []))}")
        print(f"  Categories:")
        for cat, count in sorted(s.get("categories", {}).items()):
            print(f"    {cat}: {count}")
        print(f"  Geodesic distance: "
              f"avg={s.get('geodesic_avg', -1):.1f}m "
              f"min={s.get('geodesic_min', -1):.1f}m "
              f"max={s.get('geodesic_max', -1):.1f}m")

    print(f"\n  TOTAL: {total_all} episodes across {len(stats)} splits")


def validate_splits(splits_dir: Path) -> bool:
    """Validate existing splits for common issues."""
    all_ok = True

    # Auto-detect split names from files on disk
    ep_files = sorted(splits_dir.glob("episodes_*.json.gz"))
    for ep_path in ep_files:
        split_name = ep_path.stem.replace("episodes_", "").replace(".json", "")
        if not ep_path.exists():
            print(f"  ✗ Missing: {ep_path}")
            all_ok = False
            continue

        with gzip.open(ep_path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        episodes = data["episodes"]
        print(f"  [{split_name}] {len(episodes)} episodes")

        # Check episode IDs are unique
        ids = [ep["episode_id"] for ep in episodes]
        if len(ids) != len(set(ids)):
            print(f"    ✗ Duplicate episode_ids!")
            all_ok = False

        # Check all episodes have geodesic_distance
        no_geo = [ep["episode_id"] for ep in episodes
                  if ep.get("info", {}).get("geodesic_distance", -1) <= 0]
        if no_geo:
            print(f"    ⚠  {len(no_geo)} episodes missing geodesic_distance")

        # Check scene consistency
        scenes = set(ep.get("scene_name", "?") for ep in episodes)
        print(f"    Scenes: {scenes}")

    # Check for scene overlap between any two splits
    ep_files = sorted(splits_dir.glob("episodes_*.json.gz"))
    scene_sets = {}
    for epf in ep_files:
        sname = epf.stem.replace("episodes_", "").replace(".json", "")
        scene_sets[sname] = _get_scenes(epf)

    names = sorted(scene_sets.keys())
    for i, a in enumerate(names):
        for b in names[i+1:]:
            overlap = scene_sets[a] & scene_sets[b]
            if overlap:
                print(f"  ✗ {a} ∩ {b}: {overlap}")
                all_ok = False

    if all_ok and not ep_files:
        print("  ⚠  No episode files found to validate")
        all_ok = False

    if all_ok:
        print("  ✓ All checks passed")
    return all_ok


def _get_scenes(ep_path: Path) -> Set[str]:
    """Read episode file and return set of scene names."""
    if not ep_path.exists():
        return set()
    with gzip.open(ep_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return set(ep.get("scene_name", "?") for ep in data["episodes"])


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate per-scene episodes into train/val/test splits"
    )
    parser.add_argument("--build_manifest", type=str,
                       help="Path to build_manifest.json from build_all.py")
    parser.add_argument("--scenes_dir", type=str,
                       help="Directory containing per-scene output dirs")
    parser.add_argument("--output_dir", type=str, default="splits",
                       help="Output directory for merged splits")
    parser.add_argument("--validate", type=str,
                       help="Validate existing splits in directory")
    parser.add_argument("--category_mapping", type=str,
                       help="JSON file with category_to_task_category_id")

    args = parser.parse_args()

    if args.validate:
        ok = validate_splits(Path(args.validate))
        sys.exit(0 if ok else 1)

    if args.build_manifest:
        episodes_info = _find_episodes(Path(args.build_manifest).parent)
        # Override with manifest info
        with open(args.build_manifest) as f:
            manifest = json.load(f)
        episodes_info = []
        base_dir = Path(args.build_manifest).parent
        for entry in manifest:
            scene_name = entry["scene_name"]
            split = entry["split"]
            scene_out = base_dir / scene_name
            ep_gz = scene_out / f"{scene_name}_episodes.json.gz"
            if ep_gz.exists():
                episodes_info.append({
                    "scene_name": scene_name,
                    "split": split,
                    "episodes_path": ep_gz,
                })
    elif args.scenes_dir:
        episodes_info = _find_episodes(Path(args.scenes_dir))
    else:
        # Try default output dir
        default_dir = Path(__file__).parent / "output"
        if default_dir.exists():
            episodes_info = _find_episodes(default_dir)
        else:
            print("No scenes found. Use --scenes_dir or --build_manifest.")
            sys.exit(1)

    if not episodes_info:
        print("✗ No episode files found.")
        sys.exit(1)

    print(f"Found {len(episodes_info)} scene(s) with episodes:")
    by_split = {}
    for info in episodes_info:
        s = info["split"]
        by_split.setdefault(s, []).append(info["scene_name"])
    for split, names in sorted(by_split.items()):
        print(f"  [{split}] {', '.join(names)}")

    # Load category mapping if provided
    category_mapping = None
    if args.category_mapping:
        with open(args.category_mapping) as f:
            category_mapping = json.load(f)

    output_dir = Path(args.output_dir)
    stats = aggregate(episodes_info, output_dir, category_mapping)
    print_stats(stats)

    # Save stats
    stats_path = output_dir / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats: {stats_path}")


if __name__ == "__main__":
    main()
