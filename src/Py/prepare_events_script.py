from helpers import get_regressors
from pathlib import Path

# I prepare two kinds of events files, expanded and compact, which are explained in `get_regressors`.
# These are saved under the `bids/<<SUBJECT>>/func` folder

par_list = list(range(4, 24))
par_list.remove(19)  # the data from participant 19 is missing
home_dir = Path.home()
project_dir = f"{home_dir}/implied_motion"

for par in par_list:
    get_regressors(project_root=project_dir, subj=par, expanded=True)
    get_regressors(project_root=project_dir, subj=par, expanded=False)
