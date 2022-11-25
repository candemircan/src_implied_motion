from pathlib import Path
import os

import nibabel as nib
import numpy as np
from nilearn import image


def merge_rois(project_root: str, participant: int):

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
    
    return None

if __name__ == "__main__":
    par_list = list(range(4, 24))
    par_list.remove(19)  # the data from participant 19 is missing
    home_dir = Path.home()
    project_dir = f"{home_dir}/implied_motion"

    for par in par_list:
        merge_rois(project_root=project_dir, participant=par)