import ruptures as rpt
import numpy as np
from ruptures.base import BaseCost

from common import SegmentProperties


__DIM = 2
__NOISE_STANDARD_DEVIATION = 0.22 # In ANDI noise's standard deviation is 0.12 pixel

class MultiEstimCost(BaseCost):
    """Custom cost for exponential signals."""

    # The 2 following attributes must be specified for compatibility.
    model = ""
    min_size = 3

    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, signal):
        self.signal = signal

    def error(self, start, end):
        half_of_interval = start + int( (end - start) / 2)
        first_half = self.signal[start:half_of_interval]
        second_half = self.signal[half_of_interval:end]
        estimates_1 = [self.estimators[i](first_half) for i in range(len(self.estimators))]
        estimates_2 = [self.estimators[i](second_half) for i in range(len(self.estimators))]
        return sum(abs(estimates_1[i] - estimates_2[i]) for i in range(len(estimates_1)))

def mw_rupture_cp_detection(trajectory, custom_cost, window_width, penalty=None):
    algo = rpt.Window(width=window_width, custom_cost=custom_cost, jump=1)
    if penalty == None:
        penalty = np.log(len(trajectory)) * __DIM * (__NOISE_STANDARD_DEVIATION**2)
    algo.fit(trajectory)
    change_points = algo.predict(epsilon=penalty)
    return change_points

def simple_state_classifier(trajectory, alpha, D):
    if alpha >= 1.9
        return State.Directed
    if alpha <= 0.05 and D <= 0.05:
        return State.Immobile
    return State.Free

def analyze_trajectory(trajectory, cp_detector, alpha_regressor, D_regressor, classifier):
    change_points = [0] + cp_detector(trajectory)# + [len(trajectory)] # This is technical step, which simplify saving results
    #print("Change points: ", change_points)
    segments = []
    for i in range(1, len(change_points)):
        traj_selected = trajectory[change_points[i-1]:(change_points[i]+1)]
        segments.append(SegmentProperties(D_regressor(traj_selected),
                                          alpha_regressor(traj_selected),
                                          classifier(traj_selected),
                                          change_points[i],
                                          change_points[i] - change_points[i-1]))
    return segments


def analyze_fov(fov, cp_detector, alpha_regressor, D_regressor, classifier):
    return { i:analyze_trajectory(fov[i], cp_detector, alpha_regressor, D_regressor, classifier) for i in range(len(fov)) }