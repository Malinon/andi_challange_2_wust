import sys
import joblib 

from input_utils import get_fovs_of_experiment
from common import State
from ensemble import naive_aggregation
from segmentation import mw_cp_detection, analyze_fov, single_exceed_strategy
from estimators import estimate_with_noise_1, estimate_with_noise_3
from submission_utils import create_submission

THRASHOLD = 0.1

if __name__ == "__main__":
    def alpha_estim(x):
        _, alpha = estimate_with_noise_3(x, 10)
        return alpha
    def D_estim(x):
        D, _ = estimate_with_noise_1(x, 10)
        return D
    track = dict()
    ens = dict()
    for i in range(10):
        print("Exp ", i)
        experiment_path = "../public_dat/track_2/exp_{}".format(i)
        fovs = get_fovs_of_experiment(experiment_path)
        dummy_classifier = lambda x: State.Free
        strategy = lambda a1, a2, d1, d2: single_exceed_strategy(a1, a2, THRASHOLD) or single_exceed_strategy(d1, d2, THRASHOLD)
        cp_detector = lambda x: mw_cp_detection(x, [alpha_estim], [D_estim], 40, strategy)
        experiment_result = [analyze_fov(fovs[fov_id],  cp_detector, alpha_estim, D_estim, dummy_classifier)
                                    for fov_id in range(len(fovs))]
        joblib.dump(experiment_result, "exp_res_{}".format(i))
        ensemble_res = naive_aggregation(experiment_result)
        track[i] = experiment_result
        ens[i] = ensemble_res
    create_submission(track, ens)
