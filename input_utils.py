import pandas as pd
import os

__TRAJECTORY_ID_TAG = "traj_idx"
__TRAJECTORY_NAME_START = "trajs_fov_"
__TRAJECTORY_FILE_FORMAT = "trajs_fov_{}.csv"
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


def get_number_of_fov_in_experiment(experiment_path):
    counter = 0
    for file_name in os.listdir(experiment_path):
        if file_name.startswith(__TRAJECTORY_NAME_START):
            counter += 1
    print(counter)
    return counter

def get_fovs_of_experiment(experiment_path):
    return [read_fov(os.path.join(experiment_path,__TRAJECTORY_FILE_FORMAT.format(i)))
             for i in range(get_number_of_fov_in_experiment(experiment_path))]
