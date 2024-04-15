def single_exceed_strategy(estimates_first, estimates_second, thrashold):
    return any(abs(estimates_first[i] - estimates_second[i]) for i in range(length(estimates_first)))

def mw_cp_detection(trajectory, alpha_estimators, D_estimators, window_width, strategy):
    change_points = []
    for wn_start_point in range(len(trajectory) - window_width * 2 - 1):
        first_window = trajectory[wn_start_point:(wn_start_point + window_width)]
        second_windows = trajectory[(wn_start_point + window_width - 1):(wn_start_point + 2 * window_width)]
        alpha_estimates = [est(first_window) for est in alpha_estimators]
        D_estimates = [est(first_window) for est in D_estimators]
        if strategy(alpha_estimates_first, alpha_estimates_second, D_estimates_first, D_estimates_second):
            change_points.append(wn_start_point)
    return change_points