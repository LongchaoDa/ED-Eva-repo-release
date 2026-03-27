# ED-Eva

ED-Eva is a research release for scenario-driven evaluation of trajectory predictors in autonomous driving. It accompanies the paper `Measuring What Matters: Scenario-Driven Evaluation for Trajectory Predictors in Autonomous Driving` and packages two cleaned components:

1. Frenet Optimal Trajectory (FOT) planner demo support
2. Trajectory diversity evaluation based on GAD and bounded GAD variants

This project is intended as a reference implementation for reproducing the evaluation ideas and demo workflows used in the paper. It includes example data, notebooks, visualization scripts, and cleaned planner support files.

## Repository Structure

- `ED-Eva-repo/`: main research release directory
- `ED-Eva-repo/demo_diversity.ipynb`: notebook for reproducing the code version and paper-accurate GAD
- `ED-Eva-repo/Bounded_GAD.ipynb`: notebook for bounded GAD with linear and logistic mappings
- `ED-Eva-repo/exampleData/`: example PKL files used by the notebooks and metric scripts
- `ED-Eva-repo/proposedMetric/`: standalone visualization scripts, including `GAD_vis.py` and `KDE_vis.py`
- `ED-Eva-repo/frenet_files/`: cleaned Frenet demo files, including `fot_update_traj_dynamic_on_predTraj.py`
- `frenet_optimal_trajectory_planner/`: standalone Frenet Optimal Trajectory planner source and build outputs
- `requirements.txt`: Python dependencies for the notebooks and visualization scripts

## Quick Start

### Option 1: Docker

A ready-to-run Docker image is available here:

https://hub.docker.com/repository/docker/danielda1/ed-eva/general

The published image contains the same directory structure as this codebase, so once the container starts you can follow the same commands in this README from the project root.

```bash
docker pull danielda1/ed-eva:v1
docker run --rm -it danielda1/ed-eva:v1 /bin/bash
```

### Option 2: Local Setup

If you cannot use Docker, install the Python dependencies locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On headless machines, use a non-interactive Matplotlib backend:

```bash
export MPLBACKEND=Agg
```

## Run the Diversity Evaluation

If you want the fastest local smoke test for the metric release, run:

```bash
cd ED-Eva-repo
python3 proposedMetric/GAD_vis.py
python3 proposedMetric/KDE_vis.py
```

These scripts load the example PKL files from `ED-Eva-repo/exampleData/` and save figures under `ED-Eva-repo/proposedMetric/myUse/update/`.

You can also open the notebooks:

- `ED-Eva-repo/demo_diversity.ipynb`: compares the two example PKLs and computes paper-accurate GAD
- `ED-Eva-repo/Bounded_GAD.ipynb`: applies physically bounded GAD and reports linear and logistic normalized variants

Example inputs:

- `ED-Eva-repo/exampleData/scenario_1_output.pkl`
- `ED-Eva-repo/exampleData/scenario_1_output_imputed.pkl`

## Frenet Planner Demo Support

The planner portion of this release focuses on a cleaned Frenet Optimal Trajectory demo scenario.

Key demo file:

- `ED-Eva-repo/frenet_files/FrenetOptimalTrajectory/fot_update_traj_dynamic_on_predTraj.py`

What it does:

- Runs a Frenet Optimal Trajectory planner with static obstacles and moving agents
- Updates moving agents with a constant-velocity model at each step
- Generates short predicted trajectories for moving agents and places future obstacle boxes for planning
- Replans the ego vehicle iteratively and can optionally display or save animation output

### Recommended Workflow

Only a small subset of planner-specific files is kept under `ED-Eva-repo/frenet_files/`. The recommended workflow is to pull the upstream planner repo and apply the local demo changes listed below.

Upstream repo:

- https://github.com/fangedward/frenet-optimal-trajectory-planner.git

Local changes needed after pulling upstream:

1. In `build.sh`, comment out the `apt-get` line to avoid installing system packages from the build script.
2. In `FrenetOptimalTrajectory/fot.py`, save frames into `saves/` using a path relative to the file instead of `img/frames`.
3. In `FrenetOptimalTrajectory/fot_wrapper.py`, load `build/libFrenetOptimalTrajectory.so` using an absolute path relative to the module and raise a clear error if it is missing.
4. Copy `ED-Eva-repo/frenet_files/FrenetOptimalTrajectory/fot_update_traj_dynamic_on_predTraj.py` into the pulled planner repo. This is the demo script used in the paper release.

Artifacts you can ignore or regenerate:

- `build/`
- `FrenetOptimalTrajectory/__pycache__/`
- `output/` or `saves/`, depending on how the planner is run

The standalone planner checkout included in this workspace is under `frenet_optimal_trajectory_planner/`. A simple base planner run is:

```bash
cd frenet_optimal_trajectory_planner
python3 FrenetOptimalTrajectory/fot.py --save --thread 0
```

## Citation

If you use this project in your research, please cite:

```bibtex
@article{da2025measuring,
  title={Measuring What Matters: Scenario-Driven Evaluation for Trajectory Predictors in Autonomous Driving},
  author={Da, Longchao and Isele, David and Wei, Hua and Saroya, Manish},
  journal={arXiv preprint arXiv:2512.12211},
  year={2025}
}
```

## Acknowledgements

The planner component builds on the open-source Frenet Optimal Trajectory Planner:

- https://github.com/fangedward/frenet-optimal-trajectory-planner.git

Related references:

- M. Werling et al., `Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame`
- Project demo video: https://www.youtube.com/watch?v=Cj6tAQe7UCY
