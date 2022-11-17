# Nifti to BIDS
# Then, use the following code to format nifti data into BIDS format, which we need for preprocessing.
# It also makes people's lives easier in the future for using the same data.

from pathlib import Path
from helpers import rename_nifti

par_list = list(range(4, 24))
par_list.remove(19)  # the data from participant 19 is missing
home_dir = Path.home()
project_dir = f"{home_dir}/implied_motion"

for participant in par_list:

    rename_nifti(project_root=project_dir, subject_no=participant)
