# BioVLN: A Simulation Platform for Zero-Shot VLN in Biology Laboratories

BioVLN is a simulation platform for visual language navigation (VLN) in biology laboratories. It features a dual-entry scene generation pipeline, a three-zone operational envelope model, six baseline methods, and built-in safety metrics (MCR and VRT).

## Directory Structure

```
BioVLN/
├── platform/           # Scene generation core
│   ├── scene_generator.py    — Procedural spec generator (position/rotation/goal)
│   └── config.py             — Instrument catalog + layout templates
│
├── pipeline/           # Shared 3-step pipeline + construction
│   ├── lab_builder_blender.py — Blender headless scene assembly
│   ├── auto_config.py        — Runtime config auto-generator
│   ├── step1_semantic.py     — Vertex-level semantic annotation
│   ├── step2_navmesh.py      — Recast navigation mesh generation
│   ├── step3_episodes.py     — Episode construction (goal snap + start sampling)
│   ├── run_vlfm_v4.py        — VLFM agent (GPT-4o + GDINO + SAM)
│   ├── generate_lab.py       — GLB post-processing utilities
│   ├── interactive_viewer.py — Habitat-sim interactive scene viewer
│   └── build_diverse.py      — Batch build orchestrator
│
├── scripts/            # Evaluation + utilities
│   ├── biovln_eval.py        — ★ Standard evaluation SDK (BioVLNPolicy base class)
│   ├── run_baselines.py      — Random + Oracle agents
│   ├── run_frontier.py       — Frontier Exploration (FBE)
│   ├── run_llm_frontier.py   — LLM-Frontier (DeepSeek-chat)
│   ├── run_three_zone.py     — 3-Zone Oracle
│   ├── run_vlfm_all.py       — VLFM multi-dataset runner
│   ├── rerun_baselines_safety.py — Baseline safety metric re-runner
│   ├── run_vlfm_safety.py    — VLFM + safety tracking
│   ├── run_ablation_centroid.py — Centroid vs op-face goal ablation
│   ├── generate_templates_llm.py — LLM-assisted layout template generation
│   ├── render_instrument_closeups.py — Blender instrument front-view renders
│   └── run_vlfm_viz_safety.py — VLFM trajectory visualization
│
├── lsat/               # LSAT Annotation Toolkit (Blender addon, 13 modules)
│   ├── glb_pipeline.py       — ★ Dual-strategy goal extraction core
│   ├── scanner.py / generator.py / semantics.py / collision.py
│   └── ...
│
├── vlm_servers/        # VLM model servers (Flask)
│   ├── gdino_server.py       — Grounding DINO (port 12181)
│   ├── blip_server.py        — BLIP-2 ITM (port 12182)
│   ├── sam_server.py         — SAM segmentation (port 13183)
│   └── patch_value_map_hf.py — Patch-based BLIP-2 value map
│
└── configs/            # Example scene configurations
    ├── config_two_room.py    — Two-Room baked GLB config
    └── config_lsat_target.py — LSAT target scene config
```

## Two Entry Paths

```
Path A (Procedural):   platform/config.py
                       → platform/scene_generator.py
                       → pipeline/lab_builder_blender.py
                       → pipeline/auto_config.py
                       → pipeline/step1→2→3
                       → scripts/eval

Path B (User Import):  Blender GLB + target_* markers
                       → lsat/glb_pipeline.py (target extraction)
                       → configs/config_*.py (hand-authored)
                       → pipeline/step1→2→3
                       → scripts/eval
```

Both paths converge into the shared 3-step pipeline (semantic → navmesh → episodes), followed by evaluation using one of six navigation methods.

## Evaluation Metrics

| Metric | Abbrev | Description |
|--------|--------|-------------|
| Success Rate | SR | Fraction of episodes reaching the goal |
| SPL | SPL | Success weighted by path optimality |
| Min Clearance Radius | MCR | Closest distance to any instrument surface |
| Violation Rate at Threshold | VRT | Fraction of steps within 0.5 m of any instrument |

## Quick Start

```python
from biovln_eval import BioVLNPolicy, evaluate, Action

class MyAgent(BioVLNPolicy):
    def act(self, observation):
        # observation = {position, heading, goal_position, goal_category, step}
        return Action.MOVE_FORWARD  # 0=STOP, 1=FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT

results = evaluate(MyAgent(), split="dev", dataset="lsat_target")
print(f"SR: {results['SR']:.1%}")
```

## Environment

- Python 3.9+ (conda environment `habitat`)
- habitat-sim 0.2.5
- Blender 5.1.2 (for scene construction + LSAT)
- NVIDIA GPU (tested on RTX 4090, 24 GB)

## API Keys

VLFM and LLM-Frontier require API keys set as environment variables:
```bash
export OPENAI_API_KEY="sk-..."        # GPT-4o (VLFM value maps + detection verification)
export OPENAI_BASE_URL="https://..."   # Optional: custom API endpoint
export DEEPSEEK_API_KEY="sk-..."       # DeepSeek-chat (LLM-Frontier)
```

## Citation

If you use BioVLN in your research, please cite our paper.
