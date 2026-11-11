# BioVLN Figures

> The four figures that illustrate the platform, rendered at 300 dpi (the real-world panel at 600 dpi) for inline viewing.

---

## Fig 1 — Platform Architecture

![Platform Architecture](https://raw.githubusercontent.com/ActiveButterflies/BioVLN/master/figures/fig4.png)

Instruments and room assets feed two scene-construction paths that converge on one benchmark pipeline.
- **Left (Path B):** designer-authored GLB scenes, annotated through the LSAT goal extractor.
- **Right (Path A):** procedural generation from the instrument catalog and layout templates.
- Both paths emit the same artefacts: an annotated scene GLB, a Recast navigation mesh and episode definitions, consumed by the evaluator and the baseline runners.

---

## Fig 2 — Route-Level Safety Quantities

![Route-Level Safety](https://raw.githubusercontent.com/ActiveButterflies/BioVLN/master/figures/fig0.png)

- **Minimum clearance** is the smallest body-to-obstacle distance on the route: a single close approach sets the episode value even when the terminal pose is fine.
- **Near-obstacle exposure** aggregates the same quantity over the steps that matter, so a run can be safe on arrival and still spend time inside the margin.
- These are scored separately from terminal success, which is why two controllers that reach a valid pose equally often can differ markedly on joint success.

---

## Fig 3 — Usable-Side Navigation and the Terminal-Pose Contract

![Terminal-Pose Contract](https://raw.githubusercontent.com/ActiveButterflies/BioVLN/master/figures/fig1.png)

Using a centrifuge as the example:
- **(a)** The usable-side approach follows the control face, while another side reaches comparable proximity without an admissible terminal pose.
- **(b)** The base centre must lie inside the position region on the usable side, the heading within tolerance of the outward control-face normal (30° by default), and the 0.18 m footprint must clear the surrounding furniture.
- Two failures are marked: a centre outside the region, and a correct position pointing away from the control face.

The contract is implemented in `geometry/operational_pose_v3.py`.

---

## Fig 4 — Real-World Navigation

![Real-World Navigation](https://raw.githubusercontent.com/ActiveButterflies/BioVLN/master/figures/fig3.png)

The same trial in four stages — search, find, approach, arrival — recorded twice: (a) from an external third-person camera with the base in frame, and (b) from the robot's own first-person camera. Boxes mark the detector output along the run; the arrow gives the direction of travel.

---

## File Index

| File | Figure | Used in |
|------|--------|---------|
| `fig4.png` | Platform architecture | README overview |
| `fig0.png` | Route-level safety quantities | README, evaluation metrics |
| `fig1.png` | Usable-side navigation / terminal-pose contract | README, operating-pose contract |
| `fig3.png` | Real-world navigation | README, physical-robot demonstration |
