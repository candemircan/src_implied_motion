from helpers import merge_rois
from pathlib import Path

# remove overlapping ROIs, merge the ones we want, and rename them

par_list = list(range(4, 24))
par_list.remove(19)  # the data from participant 19 is missing
home_dir = Path.home()
project_dir = f"{home_dir}/implied_motion"

for par in par_list:
    merge_rois(project_root=project_dir, participant=par)