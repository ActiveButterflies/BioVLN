# BioVLN Paper Figures

> All 4 figures from the BioVLN paper, hosted on GitHub for direct inline viewing.

---

## Fig 1: Platform Architecture

![Platform Architecture](https://raw.githubusercontent.com/ActiveButterflies/BioVLN/master/figures/fig4.png)

The BioVLN platform architecture.
- **P1 (left):** User-authored scene import (Path B) — manually built GLB scenes annotated via LSAT are processed.
- **P2 (right):** Procedural generation (Path A) — instrument catalogs and layout templates drive automated construction.
- Both paths converge into a shared benchmark pipeline with six-method evaluation and safety metrics (MCR, VRT).

---

## Fig 2: Operational-Face Goal Navigation

![Three-Zone Model](https://raw.githubusercontent.com/ActiveButterflies/BioVLN/master/figures/fig1.png)

- **Left:** Blue solid line = correct navigation to the operational face (red point); red dashed line = incorrect (reaches incubator but not its operational surface).
- **Right:** Top-down view of the operational face.

---

## Fig 3: Scene-Level Operational-Face Goal Positions

![Scene Goals](https://raw.githubusercontent.com/ActiveButterflies/BioVLN/master/figures/fig3.png)

- **Left:** Oblique overview of goal endpoints in a single-room laboratory scene.
- **Right:** Top-down visualization per instrument, with each goal position as a colored endpoint.

---

## Fig 4: Safety Boundary Visualization

![Safety](https://raw.githubusercontent.com/ActiveButterflies/BioVLN/master/figures/fig0.png)

- **Red dashed path:** Navigation trajectory violating instrument safety boundaries (enters clearance zone).
- **Blue solid path:** Safe navigation trajectory maintaining clearance.

This figure illustrates the MCR (Minimum Clearance Radius) and VRT (Violation Rate at Threshold) safety metrics derived from the three-zone operational envelope model.

---

## File Index

| File | Paper Fig | Section | Description |
|------|-----------|---------|-------------|
| `fig4.png` | Fig. 1 | Introduction | Platform architecture (dual-entry pipeline) |
| `fig1.png` | Fig. 2 | Problem Formulation | Operational-face goal navigation + three-zone model |
| `fig3.png` | Fig. 3 | Scene Generation | Scene-level operational-face goal positions |
| `fig0.png` | Fig. 4 | Safety Analysis | Safety boundary visualization (MCR/VRT) |
