#!/usr/bin/env python3
"""
Orchestrator: Generate a complete BioVLN scene from a lab_spec.json.

Workflow:
  1. Blender Builder: lab_spec.json → scene.glb + mesh_manifest.json
     (Blender runs headless with lab_builder_blender.py)
  2. AutoConfig:      lab_spec.json + manifest → _runtime_config.py
  3. Pipeline:        step1 → step2 → step3 → step4

Usage:
  # Scan available instruments (uses trimesh, fast)
  python generate_lab.py --scan /path/to/instruments/

  # Build scene and run full pipeline from a spec
  python generate_lab.py --spec lab_spec.json [--output /output/dir/]

  # Skip pipeline steps (scene only)
  python generate_lab.py --spec lab_spec.json --scene-only

  # Run only specific pipeline steps
  python generate_lab.py --spec lab_spec.json --steps 1,2,3

Blender:
  Auto-detected from BLENDER_PATH env var, or default paths.
  Set BLENDER_PATH=/path/to/blender to override.
"""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).parent.resolve()
CONDA_ENV = os.environ.get("CONDA_ENV", "habitat")

# ─── GLB coordinate conversion ─────────────────────────────────────────────

def _swap_glb_yz(glb_path: Path):
    """Swap Y and Z axes in all mesh vertex positions of a GLB file.

    This converts a Z-up GLB (Blender native) to Y-up (habitat-sim native)
    by physically rewriting vertex data.  This is more reliable than relying
    on scene_dataset_config.json \"up\" declarations which habitat-sim may
    not apply consistently to the renderer.

    GLB structure: header(12B) + JSON_chunk + BIN_chunk
    """
    with open(glb_path, 'rb') as f:
        data = bytearray(f.read())

    if data[:4] != b'glTF':
        print("  ⚠  Not a GLB file, skipping Y↔Z swap")
        return

    # Read JSON chunk
    json_len = struct.unpack_from('<I', data, 12)[0]
    if data[16:20] != b'JSON':
        print("  ⚠  Unexpected GLB chunk order")
        return
    json_start = 20
    j = json.loads(data[json_start:json_start + json_len])

    bin_len = struct.unpack_from('<I', data, json_start + json_len)[0]
    bin_start = json_start + json_len + 8  # 4 length + 4 type

    accessors = j.get('accessors', [])
    buffer_views = j.get('bufferViews', [])
    meshes = j.get('meshes', [])

    # Collect all POSITION accessors
    positions = set()
    for m in meshes:
        for p in m.get('primitives', []):
            ai = p.get('attributes', {}).get('POSITION')
            if ai is not None:
                positions.add(ai)

    swapped = 0
    for ai in positions:
        acc = accessors[ai]
        bv_idx = acc.get('bufferView')
        if bv_idx is None:
            continue
        bv = buffer_views[bv_idx]
        offset = bin_start + bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
        count = acc['count']
        comp_type = acc['componentType']  # 5126 = FLOAT
        stride = bv.get('byteStride', 12)  # default = tightly packed vec3

        if comp_type != 5126:  # FLOAT
            continue
        if acc.get('type') != 'VEC3':
            continue

        for i in range(count):
            off = offset + i * stride
            # Read 3 floats
            x = struct.unpack_from('<f', data, off)[0]
            y = struct.unpack_from('<f', data, off + 4)[0]
            z = struct.unpack_from('<f', data, off + 8)[0]
            # Swap Y and Z: (x, y, z) → (x, z, y)
            struct.pack_into('<f', data, off, x)
            struct.pack_into('<f', data, off + 4, z)
            struct.pack_into('<f', data, off + 8, y)
        swapped += 1

    # Also swap the accessor min/max values to reflect new coordinates
    for ai in positions:
        acc = accessors[ai]
        if 'min' in acc and 'max' in acc:
            old_min = acc['min']
            old_max = acc['max']
            acc['min'] = [old_min[0], old_min[2], old_min[1]]
            acc['max'] = [old_max[0], old_max[2], old_max[1]]

    # Re-serialize JSON
    new_json = json.dumps(j, separators=(',', ':')).encode('utf-8')
    if len(new_json) > json_len:
        print(f"  ⚠  JSON grew ({len(new_json)} > {json_len}), aborting Y↔Z swap")
        return
    new_json = new_json + b' ' * (json_len - len(new_json))
    data[json_start:json_start + json_len] = new_json

    with open(glb_path, 'wb') as f:
        f.write(data)

    print(f"  ✓ Y↔Z swap: {swapped} POSITION accessor(s) converted")


# ─── Existing code ─────────────────────────────────────────────────────────

# Auto-detect Blender
def _find_blender() -> Path:
    """Find the Blender executable."""
    # 1. Environment variable
    env_path = os.environ.get("BLENDER_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # 2. Known locations
    candidates = [
        PIPELINE_DIR.parent / "blender-5.1.2-linux-x64" / "blender",
        Path("/root/gpufree-data/blender-5.1.2-linux-x64/blender"),
    ]
    for c in candidates:
        if c.exists():
            return c

    # 3. System PATH
    try:
        result = subprocess.run(["which", "blender"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip())
    except FileNotFoundError:
        pass

    return None


BLENDER_PATH = _find_blender()


def _python_cmd() -> list:
    """Return the python command to use (conda run or plain python)."""
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True, text=True, timeout=10
        )
        if CONDA_ENV in result.stdout:
            return ["conda", "run", "-n", CONDA_ENV, "python3"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ["python3"]


def run_step(step_script: str, cwd: Path = None) -> bool:
    """Run a pipeline step script. Returns True on success."""
    if cwd is None:
        cwd = PIPELINE_DIR

    script_path = PIPELINE_DIR / step_script
    if not script_path.exists():
        print(f"  ❌ Script not found: {script_path}")
        return False

    cmd = _python_cmd() + [str(script_path)]
    print(f"  ▶ Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode == 0:
        print(f"  ✓ {step_script} completed\n")
        return True
    else:
        print(f"  ✗ {step_script} failed (exit {result.returncode})\n")
        return False


def _postprocess_glb(glb_path: Path):
    """Strip glTF extensions that habitat-sim's PBR renderer doesn't support.

    Removes:
      - KHR_materials_specular (not supported by Magnum/habitat-sim)
      - KHR_texture_transform on texture references (UV transforms ignored)
    """
    with open(glb_path, 'rb') as f:
        data = f.read()

    # GLB header: magic(4) + version(4) + total_length(4) = 12 bytes
    if data[:4] != b'glTF':
        print("  ⚠  Not a valid GLB file, skipping postprocess")
        return

    # Read JSON chunk
    json_chunk_len = struct.unpack_from('<I', data, 12)[0]
    json_chunk_type = data[16:20]
    if json_chunk_type != b'JSON':
        print("  ⚠  Unexpected GLB structure, skipping postprocess")
        return

    json_start = 20
    json_bytes = data[json_start:json_start + json_chunk_len]
    j = json.loads(json_bytes)

    changes = 0

    # Remove KHR_materials_specular from all materials
    for mat in j.get('materials', []):
        ext = mat.get('extensions', {})
        if 'KHR_materials_specular' in ext:
            del ext['KHR_materials_specular']
            changes += 1
        if not ext:
            mat.pop('extensions', None)

    # Remove KHR_texture_transform from texture references
    for mat in j.get('materials', []):
        pbr = mat.get('pbrMetallicRoughness', {})
        for tex_key in ('baseColorTexture', 'metallicRoughnessTexture'):
            if tex_key in pbr:
                tex_ext = pbr[tex_key].get('extensions', {})
                if 'KHR_texture_transform' in tex_ext:
                    del tex_ext['KHR_texture_transform']
                    changes += 1
                if not tex_ext:
                    pbr[tex_key].pop('extensions', None)
        # Also check normalTexture, occlusionTexture, emissiveTexture
        for tex_key in ('normalTexture', 'occlusionTexture', 'emissiveTexture'):
            if tex_key in mat:
                tex_ext = mat[tex_key].get('extensions', {})
                if 'KHR_texture_transform' in tex_ext:
                    del tex_ext['KHR_texture_transform']
                    changes += 1
                if not tex_ext:
                    mat[tex_key].pop('extensions', None)

    if changes == 0:
        print(f"  ✓ GLB postprocess: no unsupported extensions found")
        return

    # Re-serialize JSON and rebuild GLB
    new_json = json.dumps(j, separators=(',', ':')).encode('utf-8')

    # Pad to maintain alignment (JSON chunk must be padded with spaces to
    # keep the BIN chunk at the same offset, or we rewrite it entirely)
    if len(new_json) <= json_chunk_len:
        # Pad with spaces (0x20) to maintain same chunk length
        new_json = new_json + b' ' * (json_chunk_len - len(new_json))
    else:
        print(f"  ⚠  Modified JSON too large ({len(new_json)} > {json_chunk_len}), "
              f"rewriting is not supported. Skipping postprocess.")
        return

    # Rebuild the GLB: header + JSON chunk + BIN chunk
    new_data = bytearray(data[:json_start])
    new_data.extend(new_json)
    new_data.extend(data[json_start + json_chunk_len:])

    with open(glb_path, 'wb') as f:
        f.write(new_data)

    print(f"  ✓ GLB postprocess: stripped {changes} unsupported extension(s)")


def cmd_scan(instrument_dir: str):
    """Scan instrument directory and print JSON (uses trimesh)."""
    # Still use trimesh for fast scanning
    sys.path.insert(0, str(PIPELINE_DIR))
    from lab_builder import scan_instruments
    instruments = scan_instruments(Path(instrument_dir))
    print(json.dumps(instruments, indent=2))
    print(f"\n{len(instruments)} instrument(s) found.")


def cmd_build(spec_path: str, output_dir: str = None,
              scene_only: bool = False, steps: str = None):
    """Build scene from spec and optionally run pipeline."""
    spec_path = Path(spec_path).resolve()
    if not spec_path.exists():
        print(f"❌ Spec not found: {spec_path}")
        sys.exit(1)

    with open(spec_path) as f:
        spec = json.load(f)

    scene_name = spec.get("scene_name", "generated_lab")
    if output_dir is None:
        output_dir = PIPELINE_DIR.parent / "pipeline_output"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Building scene: {scene_name}")
    print(f"  Spec:      {spec_path}")
    print(f"  Output:    {output_dir}")
    print(f"  Blender:   {BLENDER_PATH}")
    print(f"  Conda env: {CONDA_ENV}")
    print(f"{'='*60}\n")

    # ── Phase 2: Blender scene construction ──────────────────────────────
    print("── Phase 2: Building scene geometry (Blender) ──")

    if not BLENDER_PATH or not BLENDER_PATH.exists():
        print("❌ Blender not found!")
        print("   Set BLENDER_PATH=/path/to/blender or install blender-5.1.2-linux-x64")
        sys.exit(1)

    builder_script = PIPELINE_DIR / "lab_builder_blender.py"
    if not builder_script.exists():
        print(f"❌ Builder script not found: {builder_script}")
        sys.exit(1)

    cmd = [
        str(BLENDER_PATH),
        "--background",
        "--python", str(builder_script),
        "--",
        "--spec", str(spec_path),
        "--out", str(output_dir),
    ]
    print(f"  ▶ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PIPELINE_DIR))

    if result.returncode != 0:
        print(f"❌ Blender build failed (exit {result.returncode})")
        sys.exit(1)

    scene_glb = output_dir / f"{scene_name}.glb"
    manifest_path = output_dir / f"{scene_name}_manifest.json"

    if not scene_glb.exists():
        print(f"❌ Scene GLB was not created: {scene_glb}")
        sys.exit(1)

    # ── Post-process: strip unsupported extensions for habitat-sim ───
    _postprocess_glb(scene_glb)

    print(f"✓ Phase 2 complete\n")

    if scene_only:
        print("── Scene-only mode: skipping pipeline ──")
        return

    # ── Phase 3: Auto-config + Pipeline ──────────────────────────────────
    print("── Phase 3: Generating config + running pipeline ──")
    sys.path.insert(0, str(PIPELINE_DIR))
    from auto_config import write_runtime_config

    runtime_path = PIPELINE_DIR / "_runtime_config.py"
    write_runtime_config(spec_path, manifest_path, scene_glb, output_dir, runtime_path)
    print()

    # Determine which steps to run
    if steps:
        step_set = {int(s.strip()) for s in steps.split(",") if s.strip().isdigit()}
    else:
        step_set = {1, 2, 3, 4}

    # Run pipeline steps
    pipeline_steps = [
        (1, "step1_semantic.py"),
        (2, "step2_navmesh.py"),
        (3, "step3_episodes.py"),
        (4, "step4_visualize.py"),
    ]

    for step_num, script in pipeline_steps:
        if step_num in step_set:
            print(f"── Step {step_num}/4: {script} ──")
            if not run_step(script, cwd=PIPELINE_DIR):
                print(f"❌ Pipeline failed at step {step_num}")
                sys.exit(1)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"✓ Scene generation complete!")
    print(f"{'='*60}")
    print(f"\n  Output files in {output_dir}:")
    for pattern in ["*.glb", "*.navmesh", "*_episodes.json.gz",
                     "*_navmesh_map.png", "*_manifest.json"]:
        for f in sorted(output_dir.glob(pattern)):
            size = f.stat().st_size
            print(f"    {f.name:40s}  {size:>10,} bytes")
    print()


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "--scan":
        if len(sys.argv) < 3:
            print("Usage: python generate_lab.py --scan /path/to/instruments/")
            sys.exit(1)
        cmd_scan(sys.argv[2])

    elif sys.argv[1] == "--spec":
        args = sys.argv[2:]
        spec_file = args[0]
        output_dir = None
        scene_only = False
        steps = None

        i = 1
        while i < len(args):
            if args[i] == "--output" and i + 1 < len(args):
                output_dir = args[i + 1]
                i += 2
            elif args[i] == "--scene-only":
                scene_only = True
                i += 1
            elif args[i] == "--steps" and i + 1 < len(args):
                steps = args[i + 1]
                i += 2
            else:
                i += 1

        cmd_build(spec_file, output_dir, scene_only, steps)

    else:
        print(f"Unknown command: {sys.argv[1]}")
        print(__doc__)
        sys.exit(1)
