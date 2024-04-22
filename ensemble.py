import numpy as np
from common import EnsembleResult, Model

def __extract_alphas(single_trajectory_results):
    alphas = []
    for fov in single_trajectory_results:
        for traj_ind in fov:
            for segment in fov[traj_ind]:
                alphas.append(segment.alpha) 
    return alphas

def __extract_ds(single_trajectory_results):
    ds = []
    for fov in single_trajectory_results:
        for traj_ind in fov:
            for segment in fov[traj_ind]:
                ds.append(segment.K)
    return ds

def naive_aggregation(single_trajectory_results):
    # TDOD: This ensmble really needs to be replaced
    alphas = __extract_alphas(single_trajectory_results)
    alpha_mean = np.mean(alphas)
    alpha_var = np.var(alphas)
    ds = __extract_ds(single_trajectory_results)
    d_mean = np.mean(ds)
    d_var = np.var(ds)
    return EnsembleResult(Model.SINGLE_STATE, [alpha_mean], [alpha_var], [d_mean], [d_var], [1.0])

def simple_aggregation(results):
    number_of_change_points = get_cp_number(results)
    if number_of_change_points == 1:
        return naive_aggregation(results)
    

    # TODO: This method also should be upgraded/replaced in the future
