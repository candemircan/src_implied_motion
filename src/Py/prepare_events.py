import os
from pathlib import Path
from collections import namedtuple

from scipy.io import loadmat
import pandas as pd

# I prepare two kinds of events files, expanded and compact, which are explained in `get_regressors`.
# These are saved under the `bids/<<SUBJECT>>/func` folder


def get_regressors(
    project_root: str,  # root directory of the project
    subj: int,  # subject no
    expanded: bool,  # boolean value to determine whether to create expanded or compact regressor files
    direction_coding: dict = {
        1: "forward",
        2: "backward",
        3: "right",
        4: "left",
    },  # how different directions are coded in the .mat files
    n_dummies: int = 8,  # number of dummy images
    tr: float = 1.2,  # repetition time
):

    """
    writes events files in bids compliant tsv format

    """
    sub_str = f"sub-{str(subj).zfill(2)}"
    file_dir = f"{project_root}/data/behavioural/{sub_str}"

    # get files
    all_files = os.listdir(file_dir)
    # keep the ones we are interested in (i.e. those that contain "Train" or "Test")
    legal_files = [file for file in all_files if ("Train" in file) or ("Test" in file)]
    # sort them as regressors will be appended in a chronological order
    legal_files.sort()

    # collect Row in rows to create a dataframe
    Row = namedtuple("Row", ["onset", "duration", "trial_type"])

    # also get the associated run file names as the names will be used
    # for tsv

    run_files = os.listdir(f"{project_root}/data/bids/{sub_str}/func/")
    run_files = sorted(run_files, key=lambda x: int(re.split(r"(\d+)", x)[-2]))
    run_files = [x.split("bold.nii")[0] for x in run_files if "bold.nii" in x]

    for file, img_file in zip(legal_files, run_files):
        complete_path = os.path.join(file_dir, file)
        mat = loadmat(complete_path)["logs"][0, 0]  # mat files have a funny format
        motion_type = "train" if "Train" in file else "test"
        rows = []

        # subtract is the starting point of scans, which is a nonzero value
        # it is used to set the first regressor at 0
        # note that it starts from the second block as the first block is to be discarded

        subtract = mat["blockOnsets"][0, 0] - n_dummies * tr

        counter = {"forward": 1, "backward": 1, "left": 1, "right": 1}

        for block in range(mat["blockOnsets"].shape[1]):
            if block == 0:  # don't record anything for the first block
                pass
            else:

                # get the direction from matching value to dictionary
                trial_type = direction_coding[mat["blockSeq"][0, block]]
                if expanded:
                    trial_type_write = (
                        f"{trial_type}_{counter[trial_type]}"
                        if motion_type == "train"
                        else f"{trial_type}_{counter[trial_type]}"
                    )
                else:
                    trial_type_write = (
                        f"{trial_type}" if motion_type == "train" else f"{trial_type}"
                    )
                counter[trial_type] += 1

                # center onsets at 0 and add previous runs' times
                onset = mat["blockOnsets"][0, block] - subtract

                # the following is to accomodate for the slightly differently
                # coded structure fields in the .mat file
                if motion_type == "train":
                    duration = (
                        mat["blockOffsets"][0, block] - mat["blockOnsets"][0, block]
                    )
                else:
                    duration = mat["bintOnset"][0, block] - mat["blockOnsets"][0, block]

                # for the df
                rows.append(
                    Row(
                        onset=onset,
                        duration=duration,
                        trial_type=trial_type_write,
                    )
                )

        df = pd.DataFrame(rows)

        if expanded:
            write_path = (
                f"{project_root}/data/bids/{sub_str}/func/expanded_{img_file}events.tsv"
            )
        else:
            write_path = f"{project_root}/data/bids/{sub_str}/func/{img_file}events.tsv"

        df.to_csv(write_path, index=False, sep="\t")

    return None


if __name__ == "__main__":

    par_list = list(range(4, 24))
    par_list.remove(19)  # the data from participant 19 is missing
    home_dir = Path.home()
    project_dir = f"{home_dir}/implied_motion"

    for par in par_list:
        get_regressors(project_root=project_dir, subj=par, expanded=True)
        get_regressors(project_root=project_dir, subj=par, expanded=False)
