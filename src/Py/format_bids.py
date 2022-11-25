import os
import datetime
import json
from pathlib import Path


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

    return None


if __name__ == "__main__":
    par_list = list(range(4, 24))
    par_list.remove(19)  # the data from participant 19 is missing
    home_dir = Path.home()
    project_dir = f"{home_dir}/implied_motion"

    for participant in par_list:

        rename_nifti(project_root=project_dir, subject_no=participant)
