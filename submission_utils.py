import functools
import os
import shutil
import sys
import zipfile

__TRACK_TAG = "track_2"
__EXPERIMENT_PREFIX = "exp_"
__RESULT_FORMAT = "fov_{}.txt"
__SUB_WORKING_DIR = "sub_temp"
__SEPARATOR = ", "
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

def create_submission(experiments):
    if os.path.isdir(__SUB_WORKING_DIR):
        shutil.rmtree(__SUB_WORKING_DIR)
    os.mkdir(__SUB_WORKING_DIR)
    TRACK_DIR_PATH = os.path.join(__SUB_WORKING_DIR, __TRACK_TAG)
    os.mkdir(TRACK_DIR_PATH)
    for exp_id in experiments.keys():
        exp_name = __EXPERIMENT_PREFIX + str(exp_id)
        exp_dir_path = os.path.join(TRACK_DIR_PATH, exp_name)
        os.mkdir(exp_dir_path)
        for fov in range(len(experiments[exp_id])):
            save_fov(exp_dir_path, fov, experiments[exp_id])
    # Zip submission
    with zipfile.ZipFile(__SUBMISSION_ZIP_NAME , 'w') as zipf:
        for root, dirs, files in os.walk(TRACK_DIR_PATH):
            for file in files:
                zipf.write(os.path.join(root, file), 
                           os.path.relpath(os.path.join(root, file), 
                                           os.path.join(TRACK_DIR_PATH, '..')))
    