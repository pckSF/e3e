from __future__ import annotations

from scs.data import TrajectoryData

################################################################################
# Hyperparameters
################################################################################
data_path: str = "data/llander_trajectories.pkl"
seed: int = 0
################################################################################

if __name__ == "__main__":
    data = TrajectoryData.load(data_path)
    print(f"Loaded trajectory data from: {data_path}")
