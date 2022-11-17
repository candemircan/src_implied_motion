#!/gpfs01/bartels/user/cdemircan/miniconda3/envs/lr_bartels/bin/python


# this is an implementation of the real x real or implied x implied
# decoding that I ended up not using, simply because the hyperparameter
# optimisation in a nested cross validation for 1000 permutations
# takes too much time (~4 days on CIN cluster for all combinations)
# and I do not have enough time after to tweak the pipeline if things
# need to change. Also the gains of this approach are unclear
# and is an empirical question we can only see by running the code.
# maybe you can try it :)

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from joblib import Parallel, delayed


from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV

from nilearn.glm.first_level import FirstLevelModel
from nilearn.maskers import NiftiMasker


parser = argparse.ArgumentParser()
parser.add_argument("--workingdir", "-w")
parser.add_argument("--sub", "-s")
parser.add_argument("--roi", "-r")
parser.add_argument("--targetdecode", "-t")
parser.add_argument("--design", "-d")
parser.add_argument("--permutations", "-p")
parser.add_argument("--condition", "-c")

args = parser.parse_args()
project_root = args.workingdir
participant = f"sub-{str(args.sub).zfill(2)}"
roi = args.roi
target_decode = args.targetdecode
design = args.design
permutations = int(args.permutations)

condition_dict = {'real':'train','implied':'test'}
condition = condition_dict[args.condition]

try:
    def nested_cross_validate(
        X,
        y,
        groups,
        decode_pipeline,
        parameter_grid
    ):
        inner_CV = LeaveOneGroupOut()
        outer_CV = LeaveOneGroupOut()
        scores = []
        y = np.array(y)

        for i,(train_index, test_index) in enumerate(outer_CV.split(X, y, groups=groups)):
            X_tr, X_tt = X[train_index,:], X[test_index,:]
            y_tr, y_tt = y[train_index], y[test_index]

            inner_groups= [x for x in groups if x!=i]

            inner_search = GridSearchCV(
                clone(decode_pipeline),
                param_grid=parameter_grid,
                scoring="accuracy",
                cv=inner_CV,
                n_jobs=-1)
            inner_search.fit(X_tr,y_tr,groups=inner_groups)

            pred = inner_search.score(X_tt,y_tt)   
            scores.append(pred)
        
        return np.array(scores).mean()


    functional_path = f"{project_root}/data/derivatives/fMRIprep/{participant}/func"

    target_event_dict = {"lr": ["left", "right"], "fb": ["forward", "backward"]}
    target_events = target_event_dict[target_decode]

    TR = 1.2

    # get file names
    func_path = f"{project_root}/data/derivatives/fMRIprep/{participant}/func"

    if design == "compact":
        event_files = glob.glob(
            f"{project_root}/data/bids/{participant}/func/{participant}*events.tsv"
        )
    elif design == "expanded":
        event_files = glob.glob(
            f"{project_root}/data/bids/{participant}/func/expanded*events.tsv"
        )

    target_files = glob.glob(
        f"{project_root}/data/bids/{participant}/func/{participant}*events.tsv"
    )
    conf_files = glob.glob(f"{func_path}/*confounds_timeseries.tsv")
    func_files = glob.glob(f"{func_path}/*T1w_desc-preproc_bold.nii.gz")

    # filter by real motion
    target_files = [x for x in target_files if condition in x]
    event_files = [x for x in event_files if condition in x]
    conf_files = [x for x in conf_files if condition in x]
    func_files = [x for x in func_files if condition in x]

    # sort them in the right order
    target_files = sorted(
        target_files, key=lambda x: int(x.split("run-")[1].split("_events")[0])
    )
    event_files = sorted(
        event_files, key=lambda x: int(x.split("run-")[1].split("_events")[0])
    )
    conf_files = sorted(
        conf_files, key=lambda x: int(x.split("run-")[1].split("_desc")[0])
    )
    func_files = sorted(
        func_files, key=lambda x: int(x.split("run-")[1].split("_space")[0])
    )

    # read the files
    targets = [pd.read_table(x) for x in target_files]
    targets = [
        x[(x["trial_type"] == target_events[0]) | (x["trial_type"] == target_events[1])]
        for x in targets
    ]

    events = [pd.read_table(x) for x in event_files]
    confs = [
        pd.read_table(x)[["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]]
        for x in conf_files
    ]

    # get the ROI mask

    roi_mask = f"{project_root}/data/derivatives/fMRIprep/{participant}/ROI/{roi}.nii"

    effects = []
    conditions_label = []
    session_label = []

    for session in range(len(func_files)):

        glm = FirstLevelModel(
            t_r=TR,
            mask_img=roi_mask,
            high_pass=0.01,
            hrf_model="spm",
            smoothing_fwhm=None,
            n_jobs=-1,
            memory=None,
        )

        # fit the glm
        glm.fit(func_files[session], events=events[session], confounds=confs[session])

        # set up contrasts: one per condition
        conditions = events[session][
            events[session]["trial_type"].str.startswith(target_events[0])
            | (events[session]["trial_type"].str.startswith(target_events[1]))
        ]["trial_type"].unique()

        # for the expanded glm targets need to repeat
        if design == "compact":
            current_targets = conditions
        elif design == "expanded":
            current_targets = []
            for cond in conditions:
                current_targets.append(cond.split("_")[0])

        # get z scored betas and labels with associated functional runs
        for target, condition in zip(current_targets, conditions):

            effects.append(glm.compute_contrast(condition, output_type="z_score"))
            conditions_label.append(target)
            session_label.append(func_files[session])

    n_runs = 8
    n_conditions = 8 if design == "expanded" else 2
    nifti_masker = NiftiMasker(
        mask_img=roi_mask,
        standardize=False,
        runs=session_label,
        smoothing_fwhm=None,
        memory_level=1,
    )

    X = nifti_masker.fit_transform(effects)
    y = np.array(conditions_label)
    groups = [[x] * n_conditions for x in range(n_runs)]
    groups = [item for items in groups for item in items]

    classifier = LogisticRegression(
        penalty="l2", C=1.0, max_iter=4000, solver="lbfgs", multi_class="auto"
    )

    decode_pipeline = Pipeline([("logistic_regression", classifier)])
    parameter_grid = {"logistic_regression__C": np.arange(0.01, 10, 0.1)}


    real_score = nested_cross_validate(
        X,
        y,
        groups,
        decode_pipeline,
        parameter_grid
    )


    permutation_array = np.loadtxt(f"{project_root}/data/utils/{design}_permutation_within.txt").astype(int)


    permutation_scores = Parallel(n_jobs=-1)\
    (delayed(nested_cross_validate)\
        (X,
        y[permutation_array[i,:]],
        groups,decode_pipeline,
        parameter_grid
        ) for i in range(permutations))

    permutation_scores = list(permutation_scores)
    permutation_scores.append(real_score)
    decode_dict = {
        "accuracy": permutation_scores,
        "type": ["permutation"] * permutations + ["real"],
        "participant": [participant] * (permutations + 1),
        "roi": [roi] * (permutations + 1),
        "design": [design] * (permutations + 1),
        "target": [target_decode] * (permutations + 1),
    }

    df = pd.DataFrame(decode_dict)

    df.to_csv(
        f"{project_root}/data/decoding/{args.condition}/{participant}_{target_decode}_{roi}_{design}.csv",
        index=False,
    )

except Exception as Argument:

    f = open(f"{project_root}/error_logs.txt", "a")
    f.write(str(Argument))
    f.close()
