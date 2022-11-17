import datetime
import glob
import json
import os
import re
from collections import namedtuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from scipy.io import loadmat

# `get_regressors` extracts regressors in the following format from the behavioural data.
#
# Make one `.tsv` file per functional run, and add them to the func folder.
#
# If `expanded==False`, a table of the following type is created:
#
# | onset | duration | motion direction |
# |-------|----------|------------------|
# | 10  | 15      | left             |
# | 30   | 15      | forward          |
# | ...   | ...      | ...              |
#
#
# If `expanded==True`, a table of this type is created:
#
# | onset | duration | motion direction |
# |-------|----------|------------------|
# | 10  | 15      | left_1             |
# | 30   | 15      | forward_1          |
# | 50 | 15 | left_2 |
# | ...   | ...      | ...              |


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
    final_wait: int = 8,  # the final wait time after the last block in a run ends
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


def rename_nifti(
    project_root: str,  # root directory of the project
    subject_no: int,  # subject no
    bold_series_size: float = 73.68783569335938,  # size of bold series images
    bold_ref_size: float = 0.281585693359375,  # size of bold reference images for some subjects
    second_bold_ref_size: float = 73.12533569335938,  # size of bold reference images for some subjects
    anat_ref_size: float = 21.750335693359375,  # size of anatomical images
    fmap_ref_size: float = 0.281585693359375,  # size of fieldmap images
):

    """renames the files `dcm2niix` converted and puts them in bids format"""

    subject_no = str(subject_no)

    # define boring references
    allowed_sizes = [
        bold_series_size,
        bold_ref_size,
        anat_ref_size,
        fmap_ref_size,
        second_bold_ref_size,
    ]
    sub_folder = f"sub-{subject_no.zfill(2)}"
    nifti_path = f"{project_root}/data/bids"
    all_imaging_data = os.listdir(f"{nifti_path}/{sub_folder}")

    # create the following folders if they don't exist already
    new_folders = [
        f"{nifti_path}/{sub_folder}/anat",
        f"{nifti_path}/{sub_folder}/func",
        f"{nifti_path}/{sub_folder}/fmap",
    ]

    for folder in new_folders:
        try:
            os.mkdir(folder)
        except FileExistsError:
            print(f"{folder} already exists. Returning now...")
            return

    ## rename anatomical image and the json

    my_anat = [
        file
        for file in all_imaging_data
        if os.path.getsize(f"{nifti_path}/{sub_folder}/{file}") / (1024 * 1024)
        == anat_ref_size
    ]
    anat_image = [
        file
        for file in all_imaging_data
        if os.path.getsize(f"{nifti_path}/{sub_folder}/{file}") / (1024 * 1024)
        == anat_ref_size
    ][0].split(".")[0]
    os.rename(
        f"{nifti_path}/{sub_folder}/{anat_image}.nii",
        f"{nifti_path}/{sub_folder}/anat/{sub_folder}_T1w.nii",
    )
    os.rename(
        f"{nifti_path}/{sub_folder}/{anat_image}.json",
        f"{nifti_path}/{sub_folder}/anat/{sub_folder}_T1w.json",
    )

    # Organise things a bit and sorted behavioural file names based on date

    beh_data = os.listdir(
        f"{project_root}/data/behavioural/{sub_folder}/"
    )  # get all files
    beh_data = [file for file in beh_data if file.endswith(".mat")]  # keep mat only
    beh_data = [x for x in beh_data if "ROILoc" not in x]  # remove the localiser run
    beh_date_dict = {
        re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}", x).group(): x for x in beh_data
    }  # create a dict where dates are the keys & file names are the values
    beh_data_sorted = sorted(
        beh_date_dict.keys(),
        key=lambda x: datetime.datetime.strptime(x, "%Y-%m-%d_%H-%M"),
    )  # get the keys in sorted format
    beh_data_sorted = [
        beh_date_dict[x] for x in beh_data_sorted
    ]  # now get the file names in a sorted way

    # Now sort nifti and json files you want to keep based on number

    nifti_data = os.listdir(f"{nifti_path}/{sub_folder}")
    nifti_bold = [x for x in nifti_data if ("bold") in x and (".nii") in x]
    nifti_bold_json = [x for x in nifti_data if ("bold") in x and (".json") in x]
    nifti_bold_sorted = sorted(
        nifti_bold, key=lambda x: int(re.split(r"(\d+)", x)[-2])
    )  # split strings by numbers, take the second item from the last as the last item is the file extension
    nifti_bold_json_sorted = sorted(
        nifti_bold_json, key=lambda x: int(re.split(r"(\d+)", x)[-2])
    )  # same as above but for json

    # Remove nifti and json files if the corresponding nifti file is not
    # something we want (due to an interrupted run or the localisation block)

    file_sizes = [
        os.stat(f"{nifti_path}/{sub_folder}/{x}").st_size / (1024 * 1024)
        for x in nifti_bold_sorted
    ]  # get sorted file sizes (in MB)
    illegal = [file_sizes.index(x) for x in file_sizes if x not in allowed_sizes]
    illegal_refs = [x - 1 for x in illegal]  # reference images for the illegal images
    illegal.extend(illegal_refs)
    nifti_bold_sorted = [
        nifti_bold_sorted[i] for i in range(len(nifti_bold_sorted)) if i not in illegal
    ]
    nifti_bold_json_sorted = [
        nifti_bold_json_sorted[i]
        for i in range(len(nifti_bold_json_sorted))
        if i not in illegal
    ]

    # now match between old names and new names you create below

    bold_dict = {x: [] for x in nifti_bold_sorted}
    json_dict = {x: [] for x in nifti_bold_json_sorted}

    run = 0
    int_counter = 0  # every run is used twice in naming .nii files, one for reference the other for bold series. This is to keep track of that
    mat_counter = 0  # need to keep seperate track of this as behavioural data file no does not match
    run_counter = 1
    for i, file in enumerate(nifti_bold_sorted):
        if int_counter == 2:  # every two files
            int_counter = 0  # new behavioural file
            mat_counter += 1  # new behavioural file
            run_counter += 1  # increase run counts

        if not int_counter:  # check type of block for new behavioral file
            if "Train" in beh_data_sorted[mat_counter]:
                block_type = "train"
            else:
                block_type = "test"

        write_run = run_counter
        image_type = "sbref" if int_counter == 0 else "bold"
        int_counter += 1

        bold_dict[
            file
        ] = f"sub-{subject_no.zfill(2)}_task-{block_type}_run-{write_run}_{image_type}"

    # now assign the proper names with extensions both for jsons and nii images
    for (key_bold, value), key_json in zip(bold_dict.items(), json_dict.keys()):
        bold_dict[key_bold] = f"{value}.nii"
        json_dict[key_json] = f"{value}.json"

    # now rename the func files
    for cur_dict in [bold_dict, json_dict]:
        for key, value in cur_dict.items():
            os.rename(
                f"{nifti_path}/{sub_folder}/{key}",
                f"{nifti_path}/{sub_folder}/func/{value}",
            )
            pass

    # add task name to json files
    for file in json_dict.values():
        side = json.load(open(f"{nifti_path}/{sub_folder}/func/{file}"))
        side["TaskName"] = "test" if "test" in file else "train"
        with open(f"{nifti_path}/{sub_folder}/func/{file}", "w") as to_write:
            json.dump(side, to_write)

    # move fieldmap to fmap folder

    e1_image = [file for file in all_imaging_data if file.endswith("e1.nii")]

    # not all participants have fieldmaps
    # carry on with the following only if they have fieldmaps
    if e1_image:
        e1_image = e1_image[0].split(".")[0]
        e2_image = [file for file in all_imaging_data if file.endswith("e2.nii")][
            0
        ].split(".")[0]
        e2_ph_image = [file for file in all_imaging_data if file.endswith("e2_ph.nii")][
            0
        ].split(".")[0]

        fmap_dict = {
            e1_image: f"{sub_folder}_magnitude1",
            e2_image: f"{sub_folder}_magnitude2",
            e2_ph_image: f"{sub_folder}_phasediff",
        }

        for key, value in fmap_dict.items():
            os.rename(
                f"{nifti_path}/{sub_folder}/{key}.nii",
                f"{nifti_path}/{sub_folder}/fmap/{value}.nii",
            )
            os.rename(
                f"{nifti_path}/{sub_folder}/{key}.json",
                f"{nifti_path}/{sub_folder}/fmap/{value}.json",
            )
            pass

        # add 'intended for' to fieldmaps
        fmap_files = os.listdir(f"{nifti_path}/{sub_folder}/fmap/")
        func_files = os.listdir(f"{nifti_path}/{sub_folder}/func/")
        func_files = [x for x in func_files if "sbref" not in x]
        func_files = [file for file in func_files if ".nii" in file]
        func_files = [f"func/{file}" for file in func_files]

        for file in fmap_files:
            if "json" in file:
                side = json.load(open(f"{nifti_path}/{sub_folder}/fmap/{file}"))
                side["IntendedFor"] = func_files
                with open((f"{nifti_path}/{sub_folder}/fmap/{file}"), "w") as to_write:
                    json.dump(side, to_write)

    else:
        os.rmdir(new_folders[2])

    # remove everything else
    for f in os.listdir(f"{nifti_path}/{sub_folder}/"):
        try:
            os.remove(os.path.join(f"{nifti_path}/{sub_folder}/", f))
        except IsADirectoryError:
            pass


# The raw files have names that do not describe the condition of the runs. After running `dcm2niix` name the runs according to BIDS. An example of the conversion of `rename_niftis` is as follows, but the parameterisation is done automatically. The necessary fields are also added to the json files and whatever is redundant is deleted. The redundancy is deleted based on file sizes (to identify interrupted runs and localiser block) and is defined based on this project. You just need to pass in the subject number to the function.
#
# *Note* : This does not include the associated `.tsv` files for the functional runs. They are to be created elsewhere.
#
# *Also note*: Some participants have fieldmaps and others don't. When available, use them. Otherwise ignore
#
# *Also also note*: The raw data structure turned out to be much more messy than anticipated. Therefore the function does not work perfectly for all participants. For some steps of the conversion, manual steps were taken (e.g. only nifti images were available for some structural scans therefore they had to be carried manually and lacked sidecars). You will just have to trust me with it or write your own scripts to go from complete raw to bids (good luck!)
#
# *Another note*: Participant 19 had incomplete data (which Gizem also reported) and is therefore excluded from the analyses
#
# ```
# sub-01
# │
# └───func
# │   │   sub-01_task-train_run-1_sbref.json
# |   |   sub-01_task-train_run-1_sbref.nii
# │   │   sub-01_task-train_run-1_bold.json
# |   |   sub-01_task-train_run-1_bold.nii
# |   |   ...
# │   │   sub-01_task-test_run-16_sbref.json
# |   |   sub-01_task-test_run-16_sbref.nii
# │   │   sub-01_task-test_run-16_bold.json
# |   |   sub-01_task-test_run-16_bold.nii
# |
# └───anat
# |   │   sub-01_T1w.json
# |   │   sub-01_T1w.nii
# |
# └───fmap
# |   |   sub-01-magnitude1.json
# |   |   sub-01-magnitude1.nii
# |   |   sub-01-magnitude2.json
# |   |   sub-01-magnitude2.nii
# |   |   sub-01-phasediff.json
# |   |   sub-01-phasediff.nii
#
# ```
#


def merge_rois(
    project_root: str,
    participant: int,
):

    """
    Merge the left and right masks and regions that we want to look at together.
    Also remove all overlapping voxels across all masks.
    """

    print(f"creating roi masks for participant {participant}")

    participant = f"sub-{str(participant).zfill(2)}"
    wang_roi = f"{project_root}/data/derivatives/wang_2015/{participant}"
    participant_roi = f"{project_root}/data/derivatives/fMRIprep/{participant}/ROI"

    if not os.path.exists(participant_roi):
        os.mkdir(participant_roi)

    ROIs = {
        "V3a": [],
        "V3b": [],
        "hV4": [],
        "LO1": [],
        "LO2": [],
        "VO1": [],
        "VO2": [],
        "hMT": [],
        "MST": [],
        "SPL1": []
    }

    groups = {
        "V1": ["V1v", "V1d"],
        "V2": ["V2v", "V2d"],
        "V3": ["V3v", "V3d"],
        "IPS_posterior": ["IPS0", "IPS1", "IPS2"],
        "IPS_anterior": ["IPS3", "IPS4", "IPS5"],
    }

    # merge left and right for the single rois
    for key in ROIs.keys():
        left_image = image.load_img(
            f"{wang_roi}/lh-{key}-roi.nii"
        )
        right_image = image.load_img(
            f"{wang_roi}/rh-{key}-roi.nii"
        )
        ROIs[key] = image.math_img("img1 + img2", img1=left_image, img2=right_image)

    # get the rois to be combined and merge left and right for all those images
    for new_img, old_imgs in groups.items():
        if new_img in ["V1", "V2", "V3"]:
            ROIs[new_img] = image.math_img(
                "img1 + img2 + img3 + img4",
                img1=f"{wang_roi}/lh-{old_imgs[0]}-roi.nii",
                img2=f"{wang_roi}/lh-{old_imgs[1]}-roi.nii",
                img3=f"{wang_roi}/rh-{old_imgs[0]}-roi.nii",
                img4=f"{wang_roi}/rh-{old_imgs[1]}-roi.nii",
            )
        else:
            ROIs[new_img] = image.math_img(
                "img1 + img2 + img3 + img4 + img5 + img6",
                img1=f"{wang_roi}/lh-{old_imgs[0]}-roi.nii",
                img2=f"{wang_roi}/lh-{old_imgs[1]}-roi.nii",
                img3=f"{wang_roi}/lh-{old_imgs[2]}-roi.nii",
                img4=f"{wang_roi}/rh-{old_imgs[0]}-roi.nii",
                img5=f"{wang_roi}/rh-{old_imgs[1]}-roi.nii",
                img6=f"{wang_roi}/rh-{old_imgs[2]}-roi.nii",
            )

    # now remove overlapping voxels in all images
    ROI_list = list(ROIs.values())
    no_rois = len(ROI_list)
    for i, key in enumerate(ROIs.keys()):

        # take img of interest
        main_img = ROI_list[i]

        compare_img = np.logical_xor.reduce([x.get_fdata() for x in ROI_list])
        new_roi_array = np.logical_and(main_img.get_fdata(), compare_img).astype(float)
        ROIs[key] = nib.Nifti1Image(new_roi_array, main_img.affine)

    # save the masks
    for key, value in ROIs.items():
        nib.save(value, f"{participant_roi}/{key}.nii")