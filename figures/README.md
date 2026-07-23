# BioVLN Paper Figures

This directory contains all figures used in the BioVLN paper. Each figure is described below with its role in the paper narrative.

---

## Fig 0: Safety Boundary Visualization
**File:** `fig0.png`  
**Paper location:** Fig. 7 (Safety Analysis)  
**Description:** Top-down view of a single-room laboratory scene showing instrument safety boundaries. Each instrument is surrounded by a circular safety radius. The **red dashed path** indicates a navigation trajectory that violates safety boundaries (enters the instrument clearance zone). The **blue solid path** shows a safe navigation trajectory maintaining clearance. This figure directly illustrates the MCR (Minimum Clearance Radius) and VRT (Violation Rate at Threshold) safety metrics.

---

## Fig 1: Three-Zone Operational Envelope
**File:** `fig1.png`  
**Paper location:** Fig. 2 (Problem Formulation)  
**Description:**  
- **Left:** Schematic of operational-face goal navigation. The **blue solid path ending at a red point** represents correct navigation — the agent reaches the incubator's operational face. The **red dashed path ending at a blue point** is incorrect — reaching the incubator's body but not facing its operational surface.  
- **Right:** Top-down view of the operational face, marked by a red solid line, showing the correct approach direction.

---

## Fig 3: Scene-Level Operational-Face Goal Positions
**File:** `fig3.png`  
**Paper location:** Fig. 5 (Scene Generation Pipeline)  
**Description:**  
- **Left:** Oblique overview of a single-room laboratory scene showing goal endpoints for all instruments.  
- **Right:** Top-down visualization showing each instrument's operational-face goal position as a colored endpoint. Color coding maps instrument categories to distinct visual markers.

---

## Fig 4: Platform Architecture
**File:** `fig4.png`  
**Paper location:** Fig. 1 (Platform Architecture)  
**Description:** Complete BioVLN platform architecture showing:
- **P1 (left):** User-authored scene import (Path B) — manually built GLB scenes with artist-placed `target_*` markers are processed through LSAT goal extraction.
- **P2 (right):** Procedural generation (Path A) — instrument catalogs and layout templates drive automated scene construction.
- Both paths converge into a shared three-step pipeline (semantic annotation → navigation mesh → episode construction) and a six-method evaluation layer with safety metrics (MCR, VRT).

---

## Fig: Three-Zone Model
**Files:** `three_zone_model.png`, `three_zone_delta.png`, `operational_envelope.png`, `scene_operational_envelope.png`  
**Paper location:** Fig. 2, Fig. 5 (Problem Formulation, Scene Generation)  
**Description:**  
- `three_zone_model.png` — Illustrates the three nested spatial zones around each instrument: Zone 1 (physical bounding box), Zone 2 (0.25 m collision clearance buffer), Zone 3 (operational envelope extending from the operational face).
- `three_zone_delta.png` — Shows the delta parameter δ_i (0.5–0.9 m) that defines Zone 3 depth for each instrument category.
- `operational_envelope.png` — Generic operational envelope diagram showing approach direction and success radius.
- `scene_operational_envelope.png` — Scene-level rendering with all instrument envelopes overlaid.

---

## Fig: Per-Category Performance
**Files:** `per_category_breakdown.png`, `category_heatmap.png`  
**Paper location:** Table 7, Fig. 6 (Experiments)  
**Description:**  
- `per_category_breakdown.png` — Bar chart showing Frontier Exploration success rate broken down by instrument category across three datasets (MS-DEV, LSAT, Two-Room).
- `category_heatmap.png` — Heatmap visualization of success rate across methods × instrument categories, highlighting VLFM's domain gap with laboratory instruments.

---

## Fig: Difficulty Analysis
**File:** `difficulty_breakdown.png`  
**Paper location:** Section 5.2 (Experiments)  
**Description:** Bar chart showing success rate stratified by geodesic difficulty tier (easy: <3.0 m, medium: 3.0–6.0 m, hard: >6.0 m) for Frontier Exploration and 3-Zone Oracle on Multi-Scene splits.

---

## Fig: Failure Analysis
**File:** `failure_analysis.png`  
**Paper location:** Section 5.5 (VLM Analysis)  
**Description:** Visualization of VLFM failure modes, categorized by root cause: surface texture sparsity, VLM recognition error, navigation timeout, and collision. Shows the distribution of failure types across datasets.

---

## Fig: Method Comparison
**File:** `paper_comparison_chart.png`  
**Paper location:** Table 5 summary (Experiments)  
**Description:** Overall success rate comparison across all 6 methods on Multi-Scene DEV. Horizontal bars with annotated percentages show the performance hierarchy from Random (26.9%) through 3-Zone Oracle (83.8%).

---

## Fig: Pipeline Architecture Detail
**File:** `pipeline_architecture.png`  
**Paper location:** Fig. 1 detail (BioVLN Platform)  
**Description:** Detailed block diagram of the dual-entry pipeline: Path A (procedural) and Path B (user-authored), with component-level detail for each of the three shared pipeline steps.

---

## Fig: Scene Layout
**File:** `scene_layout.png`  
**Paper location:** Fig. 5 (Scene Generation)  
**Description:** Visual comparison of the four layout templates defined in BioVLN: single-room (8×6 m, implemented), two-room (6×5 m + 3.5×3.5 m, implemented), suite, and open-plan (defined for future expansion).

---

## Fig: Trajectory Visualizations
**Files:** `lsat_trajectories.png`, `two_room_trajectories.png`  
**Paper location:** Appendix (Supplementary Material)  
**Description:**  
- `lsat_trajectories.png` — Top-down trajectory overlay on the LSAT Target scene, showing Frontier Exploration paths with color-coded success/failure outcomes.  
- `two_room_trajectories.png` — Corresponding trajectory visualization for the Two-Room benchmark, highlighting cross-room navigation paths through the 1.5 m central corridor.
