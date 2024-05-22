import sys
import joblib

import model
from input_utils import get_fovs_of_experiment
from common import State
from ensemble import naive_aggregation, simple_aggregation
from segmentation import analyze_fov, MultiEstimCost, mw_rupture_cp_detection, simple_state_classifier
from estimators import estimate_with_noise_1, estimate_with_noise_3
from _03_characteristics import get_characteristics_new
from submission_utils import create_submission
from cp_detector import CPDetector

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from multiprocessing import Pool

import src.deep.tf_dain

THRASHOLD = 0.1

__WEIGHTS_PATH = "../model.weights.h5"
__GRADIENT_BOOSTING_PATH = "../model_alpha_boost"
__GRADIENT_BOOSTING_D_PATH = "../model_d_boost"

if __name__ == "__main__":
    def alpha_estim(x):
        _, alpha = estimate_with_noise_3(x, 10)
        return alpha
    def D_estim(x):
        if (len(x) < 10):
            print("Too short trajectory")
        D, _ = estimate_with_noise_2(x, 10)
        return D
    track = dict()
    ens = dict()
    class_model = model.generate_model()
    class_model.load_weights(__WEIGHTS_PATH)
    def run_experiment(i):
        gradient_boosting = joblib.load(__GRADIENT_BOOSTING_PATH)
        def alpha_estim_boost(x):
            #print("Estimating alpha ", x.shape)
            return gradient_boosting.predict(get_characteristics_new(x).fillna(0))[0]
        gradient_boosting_d = joblib.load(__GRADIENT_BOOSTING_D_PATH)
        def d_estim_boost(x):
            return gradient_boosting_d.predict(get_characteristics_new(x).fillna(0))[0]
        experiment_path = "../public_data_validation_v1/track_2/exp_{}".format(i)
        fovs = get_fovs_of_experiment(experiment_path)
        dummy_classifier = lambda x: State.Free
        #cost = SingleEstimCost(alpha_estim)
        cost = MultiEstimCost([alpha_estim, D_estim])
        #cp_detector = lambda x: mw_rupture_cp_detection(x, cost, 60)
        cp_detector = CPDetector("../changepoint_model.keras")
        experiment_result = [analyze_fov(fovs[fov_id],  cp_detector, alpha_estim_boost, d_estim_boost, simple_state_classifier)
                                    for fov_id in range(len(fovs))]
        joblib.dump(experiment_result, "exp_res_{}".format(i))
        experiment_result = joblib.load("exp_res_{}".format(i))
        ensemble_res = simple_aggregation(experiment_result, class_model, fovs)
        track[i] = experiment_result
        ens[i] = ensemble_res
        print("Exp FInished ", i)
    """for i in range(13):
        experiment_path = "../public_data_validation_v1/track_2/exp_{}".format(i)
        fovs = get_fovs_of_experiment(experiment_path)
        experiment_result = joblib.load("exp_res_{}".format(i))
        ensemble_res = simple_aggregation(experiment_result, class_model, fovs)
        track[i] = experiment_result
        ens[i] = ensemble_res"""

    #with Pool(13) as pool:
    #    pool.map(run_experiment, range(13))
    for i in range(13):
        run_experiment(i)
    create_submission(track, ens)
