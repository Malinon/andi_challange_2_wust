import pandas as pd

__TRAJECTORY_ID_TAG = "traj_idx"
__X_TAG = "x"
__Y_TAG = "y"

def read_fov(file_path):
    df = pd.read_csv(file_path)
    number_of_trajectories = int(max(df[__TRAJECTORY_ID_TAG]))
    trajectories = []
    for traj_id in range(number_of_trajectories):
        mask = df.traj_idx == traj_id
        traj = df[mask][[__X_TAG, __Y_TAG]].values
        trajectories.append(traj)
    return trajectories
