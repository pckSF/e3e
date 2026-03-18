from __future__ import annotations

from scs.data import TrajectoryData

################################################################################
# Hyperparameters
################################################################################
data_path: str = "data/llander_trajectories/episodes_1000_maxlen_250.hdf5"
seed: int = 0
################################################################################

if __name__ == "__main__":
    data = TrajectoryData.load_hdf5(data_path)
    print(f"Loaded trajectory data from: {data_path}")
