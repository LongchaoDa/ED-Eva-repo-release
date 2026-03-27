# ED-Eva: Planner and Diversity Evaluation Release

This release contains two cleaned components for reproducible research:

1) **Frenet Optimal Trajectory Planner (cleaned)**
2) **Trajectory diversity evaluation (GAD + bounded variants)**

Both are provided as reference implementations for the paper:
`Measuring What Matters: Scenario-Driven Evaluation for Trajectory Predictors in Autonomous Driving`.

---

## What is included

### 1) Frenet planner demo support
Location: `frenet_files/`

Key file for the demo scenario:
- `frenet_files/FrenetOptimalTrajectory/fot_update_traj_dynamic_on_predTraj.py`

What it does:
- Runs a Frenet Optimal Trajectory (FOT) planner with static obstacles and moving agents.
- Moving agents are updated each step with a constant velocity model.
- For each moving agent, a short predicted trajectory is generated and used to place a *future* obstacle box for planning.
- The ego vehicle replans iteratively and advances along the selected trajectory.
- Optional animation/GIF output is supported.

#### Recommended workflow: pull upstream repo + apply local changes
Only a small subset of files are kept locally under `frenet_files/`. You should
pull the official repo and apply the small set of local changes listed below.

Upstream repo:
- https://github.com/fangedward/frenet-optimal-trajectory-planner.git

Local changes needed after pulling upstream:
1) `build.sh`: comment out the `apt-get` line (avoid installing system packages in scripts).
2) `FrenetOptimalTrajectory/fot.py`: save frames into `saves/` using a path relative to the file instead of `img/frames`.
3) `FrenetOptimalTrajectory/fot_wrapper.py`: load `build/libFrenetOptimalTrajectory.so` via an absolute path relative to the module, and raise a clear error if missing.
4) Copy `frenet_files/FrenetOptimalTrajectory/fot_update_traj_dynamic_on_predTraj.py` into the pulled repo (this is the demo script used by the paper).

Local-only artifacts you can ignore or regenerate in the pulled repo:
- `build/` (CMake outputs)
- `FrenetOptimalTrajectory/__pycache__/`
- `saves/` (demo GIFs/frames; keep only if you want the media)

### 2) Diversity evaluation (GAD + bounded GAD)
Location: root notebooks
- `demo_diversity.ipynb` (reproduces the code version and the paper-accurate GAD)
- `Bounded_GAD.ipynb` (adds physical bounding for GAD, with linear and logistic mappings)

Example data (two PKLs):
- `exampleData/scenario_1_output.pkl`
- `exampleData/scenario_1_output_imputed.pkl`

---

## How to run the planner demo

From the root directory:

```bash
python3 frenet_optimal_trajectory_planner/FrenetOptimalTrajectory/fot_update_traj_dynamic_on_predTraj.py -d -s
```

Flags:
- `-d` / `--display`: show animation
- `-s` / `--save`: save a GIF in `frenet_optimal_trajectory_planner/saves/`
- `-v` / `--verbose`: print extra state info
- `-t` / `--thread`: number of threads (default: 0)

---

## How to run diversity evaluation

From the root directory, open the notebooks:

- `demo_diversity.ipynb`: compares the two example PKLs and computes paper-accurate GAD.
- `Bounded_GAD.ipynb`: applies physical bounding and reports both linear and logistic normalized GAD.

Both notebooks use the example PKLs under `exampleData/` and save figures into the project root by default.

---

## Acknowledgements

The planner implementation builds upon the open-source Frenet Optimal Trajectory Planner by fangedward:
- https://github.com/erdos-project/frenet_optimal_trajectory_planner.git

Original references from that project:
- M. Werling et al., *Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame*.
- Project demo video: https://www.youtube.com/watch?v=Cj6tAQe7UCY

If you find this release useful, please cite our paper:

```
@article{da2025measuring,
  title={Measuring What Matters: Scenario-Driven Evaluation for Trajectory Predictors in Autonomous Driving},
  author={Da, Longchao and Isele, David and Wei, Hua and Saroya, Manish},
  journal={arXiv preprint arXiv:2512.12211},
  year={2025}
}
```
