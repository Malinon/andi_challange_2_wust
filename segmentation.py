from common import SegmentProperties

def single_exceed_strategy(estimates_first, estimates_second, thrashold):
    return any(abs(estimates_first[i] - estimates_second[i]) for i in range(len(estimates_first)))

def mw_cp_detection(trajectory, alpha_estimators, D_estimators, window_width, strategy):
    change_points = [0]
    traj_len = len(trajectory)
    for wn_start_point in range(traj_len - window_width * 2 - 1):
        first_window = trajectory[wn_start_point:(wn_start_point + window_width)]
        second_windows = trajectory[(wn_start_point + window_width - 1):(wn_start_point + 2 * window_width)]
        alpha_estimates_first = [est(first_window) for est in alpha_estimators]
        D_estimates_first = [est(first_window) for est in D_estimators]
        alpha_estimates_second = [est(first_window) for est in alpha_estimators]
        D_estimates_second = [est(first_window) for est in D_estimators]
        if strategy(alpha_estimates_first, alpha_estimates_second, D_estimates_first, D_estimates_second):
            change_points.append(wn_start_point)
    change_points.append(traj_len)
    return change_points

def analyze_trajectory(trajectory, cp_detector, alpha_regressor, D_regressor, classifier):
    change_points = cp_detector(trajectory)
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