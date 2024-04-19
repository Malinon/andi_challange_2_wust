import functools
import os
import shutil
import sys
import zipfile

__TRACK_TAG = "track_2"
__EXPERIMENT_PREFIX = "exp_"
__RESULT_FORMAT = "fov_{}.txt"
__ENSEMBLE_FORMAT = 
__ENS_HEADER_FORMAT = "model: {}; num_state: {}"
__SUB_WORKING_DIR = "sub_temp"
__SEPARATOR = ", "
__ENS_SEPERATOR = '; '
__SUBMISSION_ZIP_NAME = "submission.zip"

def save_fov(dir_path, fov, experiment):
    fov_results = experiment[fov]
    output_name = os.path.join(dir_path, __RESULT_FORMAT.format(fov))
    with open(output_name, 'a') as f:
        for traj_id in fov_results.keys():  
            f.write(str(traj_id))
            f.write(__SEPARATOR)
            seg_desc = functools.reduce(lambda a, b: a + __SEPARATOR + b, (str(seg) for seg in fov_results[traj_id]))
            f.write(seg_desc)
            f.write("\n")

def save_ensemble(dir_path, experiment_id, ensemble_results):
    ens_result = ensemble_results[experiment_id]
    states_num = get_number_of_states(ens_result)
    os.path.join(dir_path, __ENSEMBLE_FORMAT.format(fov))
    with open(output_name, 'a') as f:
        f.write(__ENS_HEADER_FORMAT.format(ens_result.model.value, states_num))
        f.write("\n")
        a_mean_desc = functools.reduce(lambda a, b: a + __ENS_SEPARATOR + b, (str(a_mean) for a_mean in  ens_result.alphas_mean))
        f.write(a_mean_desc + "\n")
        a_var_desc = functools.reduce(lambda a, b: a + __ENS_SEPARATOR + b, (str(a_mean) for a_mean in  ens_result.alphas_var))
        f.write(a_var_desc + "\n")
        d_mean_desc = functools.reduce(lambda a, b: a + __ENS_SEPARATOR + b, (str(d_mean) for d_mean in  ens_result.d_mean))
        f.write(d_mean_desc + "\n")
        d_var_desc = functools.reduce(lambda a, b: a + __ENS_SEPARATOR + b, (str(d_mean) for d_mean in  ens_result.d_var))
        f.write(d_var_desc + "\n")
        weights_desc = functools.reduce(lambda a, b: a + __ENS_SEPARATOR + b, (str(w) for w in  ens_result.weights))
        f.write(weights_desc)


def create_submission(experiments, model, alphas_mean, alphas_var, d_mean, d_alpha, weights):
    if os.path.isdir(__SUB_WORKING_DIR):
        shutil.rmtree(__SUB_WORKING_DIR)
    os.mkdir(__SUB_WORKING_DIR)
    TRACK_DIR_PATH = os.path.join(__SUB_WORKING_DIR, __TRACK_TAG)
    os.mkdir(TRACK_DIR_PATH)

    for exp_id in experiments.keys():
        exp_name = __EXPERIMENT_PREFIX + str(exp_id)
        exp_dir_path = os.path.join(TRACK_DIR_PATH, exp_name)
        os.mkdir(exp_dir_path)
        save_ensemble(exp_dir_path, experiment_id, ensemble_results)
        for fov in range(len(experiments[exp_id])):
            save_fov(exp_dir_path, fov, experiments[exp_id])

    # Zip submission
    with zipfile.ZipFile(__SUBMISSION_ZIP_NAME , 'w') as zipf:
        for root, dirs, files in os.walk(TRACK_DIR_PATH):
            for file in files:
                zipf.write(os.path.join(root, file), 
                           os.path.relpath(os.path.join(root, file), 
                                           os.path.join(TRACK_DIR_PATH, '..')))
    