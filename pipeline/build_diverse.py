#!/usr/bin/env python3
"""
Build diverse BioVLN scenes from layout templates.

Workflow:
  1. Generate specs via scene_generator.py
  2. For each spec: Blender build → semantic → navmesh → episodes
  3. Post-process all scenes (geodesic + difficulty)
  4. Aggregate into splits

Usage:
  # Dry run: just generate specs and show what would be built
  python build_diverse.py --dry-run

  # Build single scene from a spec
  python build_diverse.py --spec specs/lab_single_room_s42_v0_spec.json

  # Build MVP batch (6 scenes: single_room × 4 + open_plan × 2)
  python build_diverse.py --mvp

  # Build full batch (12 scenes: all 4 templates × 3 variations)
  python build_diverse.py --full

  # Resume from last checkpoint (skip already-built scenes)
  python build_diverse.py --mvp --resume
"""

import json, os, sys, subprocess, hashlib, time
from pathlib import Path
from collections import Counter

# ─── Paths ────────────────────────────────────────────────────────────────
MULTI_SCENE = Path(__file__).parent.resolve()
PIPELINE_DIR = Path("/root/gpufree-data/pipeline")
BLENDER_PATH = Path("/root/gpufree-data/blender-5.1.2-linux-x64/blender")
CONDA_PYTHON = ["conda", "run", "-n", "habitat", "python3"]
BUILDER_SCRIPT = PIPELINE_DIR / "lab_builder_blender.py"

OUTPUT_ROOT = MULTI_SCENE / "output"
SPECS_DIR = MULTI_SCENE / "specs"
SPLITS_DIR = OUTPUT_ROOT / "splits"
CHECKPOINT_FILE = MULTI_SCENE / ".build_diverse_checkpoint.json"


# ─── Helpers ────────────────────────────────────────────────────────────────

def scene_name_to_seed(name: str) -> int:
    """Deterministic random_seed from scene name (MD5 hash)."""
    h = hashlib.md5(name.encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def run_cmd(cmd, cwd=None, timeout=600):
    """Run a command, stream output, return (success, elapsed_seconds)."""
    t0 = time.time()
    desc = " ".join(cmd) if isinstance(cmd, list) else cmd
    short = desc[:80] + ("..." if len(desc) > 80 else "")
    print(f"  ▶ {short}")

    try:
        result = subprocess.run(
            cmd, cwd=cwd,
            capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"    ✗ FAILED ({elapsed:.0f}s)")
            stderr_tail = result.stderr.strip().split("\n")[-5:]
            for line in stderr_tail:
                print(f"    [stderr] {line}")
            return False, elapsed
        print(f"    ✓ ({elapsed:.0f}s)")
        return True, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"    ✗ TIMEOUT after {elapsed:.0f}s")
        return False, elapsed


def generate_specs(mvp=False) -> list:
    """Generate specs using scene_generator.py. Returns list of spec dicts."""
    sys.path.insert(0, str(MULTI_SCENE))
    from config import SPLIT_DISTRIBUTION, SPLIT_DISTRIBUTION_MVP
    from scene_generator import generate_all_for_split

    dist = SPLIT_DISTRIBUTION_MVP if mvp else SPLIT_DISTRIBUTION
    specs = []
    for split in ["train", "val", "test"]:
        specs.extend(generate_all_for_split(split, mvp=mvp))
    return specs


def save_checkpoint(built_scenes: list):
    """Save list of successfully built scene names."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(built_scenes, f)


def load_checkpoint() -> list:
    """Load list of already-built scene names."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return []


# ─── Build Pipeline ────────────────────────────────────────────────────────

def build_scene(spec_path: Path) -> bool:
    """Build one scene: Blender → semantic → navmesh → episodes. Returns True on success."""
    with open(spec_path) as f:
        spec = json.load(f)

    scene_name = spec["scene_name"]
    scene_dir = OUTPUT_ROOT / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    spec_file = SPECS_DIR / f"{scene_name}_spec.json"

    print(f"\n{'='*60}")
    print(f"Building: {scene_name}")
    print(f"  Layout: {spec.get('layout_type', '?')}")
    print(f"  Split:  {spec.get('split', '?')}")
    print(f"  Instruments: {len(spec.get('instruments', []))}")
    print(f"{'='*60}")

    # ── Step 0: Blender build ──
    print("\n── [0/4] Blender scene construction ──")
    blender_cmd = [
        str(BLENDER_PATH),
        "--background",
        "--python", str(BUILDER_SCRIPT),
        "--",
        "--spec", str(spec_file),
        "--out", str(scene_dir),
    ]
    ok, _ = run_cmd(blender_cmd, cwd=str(PIPELINE_DIR), timeout=300)
    if not ok:
        return False

    glb_path = scene_dir / f"{scene_name}.glb"
    manifest_path = scene_dir / f"{scene_name}_manifest.json"
    if not glb_path.exists():
        print(f"  ✗ GLB not created: {glb_path}")
        return False

    # GLB post-process (strip extensions only)
    sys.path.insert(0, str(PIPELINE_DIR))
    from generate_lab import _postprocess_glb
    _postprocess_glb(glb_path)

    # Copy spec to scene dir for reference
    import shutil
    shutil.copy2(spec_file, scene_dir / f"{scene_name}_spec.json")

    # ── Step 1: Auto-config (writes pipeline/_runtime_config.py) ──
    print("\n── [1/4] Auto-config ──")
    pipeline_config = PIPELINE_DIR / "_runtime_config.py"
    auto_cfg_cmd = CONDA_PYTHON + [
        str(PIPELINE_DIR / "auto_config.py"),
        "--spec", str(spec_file),
        "--manifest", str(manifest_path),
        "--scene", str(glb_path),
        "--out", str(scene_dir),
    ]
    ok, _ = run_cmd(auto_cfg_cmd, cwd=str(MULTI_SCENE), timeout=60)
    if not ok:
        print(f"  ✗ Auto-config failed")
        return False

    # Save scene_dataset_config.json
    scene_cfg = {
        "stages": {
            "paths": {".glb": [f"{scene_name}.glb"]},
            "default_attributes": {
                "shader_type": "pbr",
                "semantic_descriptor_filename": f"{scene_name}.semantic.txt",
                "semantic_asset": f"{scene_name}.semantic.glb",
                "has_semantic_textures": True,
            }
        },
        "objects": {},
        "light_setups": {}
    }
    with open(scene_dir / "scene_dataset_config.json", "w") as f:
        json.dump(scene_cfg, f, indent=2)

    # ── Step 2-4: Pipeline steps (read from _runtime_config.py via config.py) ──
    print("\n── [2/4] Semantic annotation ──")
    ok, _ = run_cmd(
        CONDA_PYTHON + [str(PIPELINE_DIR / "step1_semantic.py")],
        cwd=str(PIPELINE_DIR), timeout=120
    )
    if not ok:
        print(f"  ⚠  Step1 semantic failed — continuing anyway")

    print("\n── [3/4] NavMesh generation ──")
    ok, _ = run_cmd(
        CONDA_PYTHON + [str(PIPELINE_DIR / "step2_navmesh.py")],
        cwd=str(PIPELINE_DIR), timeout=120
    )
    if not ok:
        print(f"  ✗ NavMesh failed")
        return False

    # ── Step 4: Episodes ──
    print("\n── [4/4] Episode generation ──")
    ok, _ = run_cmd(
        CONDA_PYTHON + [str(PIPELINE_DIR / "step3_episodes.py")],
        cwd=str(PIPELINE_DIR), timeout=300
    )
    if not ok:
        print(f"  ✗ Episode generation failed")
        return False

    print(f"\n✓ Scene '{scene_name}' built successfully!")
    return True


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build diverse BioVLN scenes")
    ap.add_argument("--dry-run", action="store_true",
                    help="Generate specs only, don't build")
    ap.add_argument("--spec", type=str,
                    help="Build a single spec")
    ap.add_argument("--mvp", action="store_true",
                    help="Build MVP batch (6 scenes)")
    ap.add_argument("--full", action="store_true",
                    help="Build full batch (12 scenes)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip already-built scenes")
    ap.add_argument("--skip-postprocess", action="store_true",
                    help="Skip post-processing")
    args = ap.parse_args()

    if not any([args.dry_run, args.spec, args.mvp, args.full]):
        ap.print_help()
        return

    # ── Generate specs ──
    if args.spec:
        spec_path = Path(args.spec)
        if not spec_path.exists():
            print(f"ERROR: Spec not found: {spec_path}")
            sys.exit(1)
        with open(spec_path) as f:
            specs = [json.load(f)]
    elif args.mvp:
        print("Generating MVP specs (6 scenes)...")
        specs = generate_specs(mvp=True)
    elif args.full:
        print("Generating full specs (12 scenes)...")
        specs = generate_specs(mvp=False)
    else:
        specs = []

    if args.dry_run:
        print(f"\n=== Dry Run: {len(specs)} specs ===")
        for spec in specs:
            sn = spec["scene_name"]
            sp = spec["split"]
            lt = spec.get("layout_type", "?")
            nc = len(spec.get("target_categories", []))
            ni = len(spec.get("instruments", []))
            room = spec["room"]
            has_mr = "rooms" in spec
            mr_str = f" | {len(spec['rooms'])} rooms" if has_mr else ""
            print(f"  {sn:<35} [{sp:<5}] {lt:<12} "
                  f"{room['width']:.1f}×{room['depth']:.1f}m "
                  f"{ni} instruments {nc} categories{mr_str}")

        by_split = {"train": [], "val": [], "test": []}
        for spec in specs:
            by_split[spec["split"]].append(spec["scene_name"])
        print(f"\nSummary:")
        for split, names in by_split.items():
            print(f"  [{split}] {len(names)} scenes")
        print("\nDry run complete. Run without --dry-run to build.")
        return

    # ── Save all specs ──
    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        sn = spec["scene_name"]
        spath = SPECS_DIR / f"{sn}_spec.json"
        with open(spath, "w") as f:
            json.dump(spec, f, indent=2)

    # ── Build each scene ──
    checkpoint = load_checkpoint() if args.resume else []
    built = []

    for i, spec in enumerate(specs):
        sn = spec["scene_name"]
        if args.resume and sn in checkpoint:
            print(f"\n[{i+1}/{len(specs)}] SKIP {sn} (already built)")
            built.append(sn)
            continue

        print(f"\n[{i+1}/{len(specs)}] {sn}")
        spath = SPECS_DIR / f"{sn}_spec.json"
        ok = build_scene(spath)
        if ok:
            built.append(sn)
            save_checkpoint(built)
        else:
            print(f"\n  ✗ FAILED: {sn} — continuing with next scene")
            # Don't break — try remaining scenes

    # ── Post-processing ──
    if not args.skip_postprocess and built:
        print(f"\n{'='*60}")
        print(f"Post-processing {len(built)} scenes...")
        print(f"{'='*60}")

        pp_cmd = CONDA_PYTHON + [
            str(MULTI_SCENE / "postprocess_episodes.py"),
            "--all",
            "--scenes_root", str(OUTPUT_ROOT),
        ]
        run_cmd(pp_cmd, cwd=str(MULTI_SCENE), timeout=600)

        print("\n── Aggregating splits ──")
        agg_cmd = CONDA_PYTHON + [
            str(MULTI_SCENE / "aggregate_splits.py"),
            "--build_manifest", str(OUTPUT_ROOT / "build_manifest.json"),
            "--output_dir", str(SPLITS_DIR),
        ]
        run_cmd(agg_cmd, cwd=str(MULTI_SCENE), timeout=300)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Build Complete!")
    print(f"{'='*60}")
    print(f"  Scenes attempted: {len(specs)}")
    print(f"  Scenes built:     {len(built)}")
    print(f"  Failed:           {len(specs) - len(built)}")

    if built:
        print(f"\n  Output directory: {OUTPUT_ROOT}/")
        print(f"  Specs:            {SPECS_DIR}/")
        total_glb = 0
        for sn in built:
            glb = OUTPUT_ROOT / sn / f"{sn}.glb"
            if glb.exists():
                total_glb += glb.stat().st_size
        print(f"  Total GLB size:   {total_glb / 1024**3:.1f} GB")

    if len(built) < len(specs):
        failed = [specs[i]["scene_name"] for i in range(len(specs))
                  if specs[i]["scene_name"] not in built]
        print(f"\n  Failed scenes: {failed}")


if __name__ == "__main__":
    main()
