from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

"""
Here I took Vincent's code exactly as is and adapted it a tiny bit to fit my data structure.
He described the permutation as follows:

Also the link to the original code:
https://github.com/PibeChorro/Magic_fMRI/blob/master/analysis/group_analysis/permutation_max_statistic.py
"""

################################
# FUNCTIONALITY OF THIS SCRIPT #
################################
# FIRST STEP
# Get all important path information and names of ROIs.
# SECOND STEP
# Iterate over ROIs and read in the null distribution for the current ROI from
# all subjects. From this huge 2D array (num_permutations x num_subjects)
# calculate the mean over subjects (=1D array with length num_permutations).
# Cast together all mean null distributions (resulting again in a 2D array
# num_permutations x num_ROIs) and get the max value for each permutation along
# the ROIs (=1D array with length num_permutations).
# The mean accuracy of the ROIs over subjects is also calculated and saved.
# This mean is later tested against the max statistic distribution.
# In between all null distributions of all subjects in one ROI are plotted in
# one figure to make sure that it is centered around chance level.
# THIRD STEP
# Calculate corrected p-value for each ROI: Look how many accuracy values in
# the max statistic distribution are higher than the observed mean accuracy
# value (called C)
# p_corrected = (C+1)/(num_permutation+1)
# FOURTH STEP
# Plot the p-value for each ROI (first figure)
# For each ROI calculate the 95% confidence interval and plot mean + CI next
# to the max statistic.
# Plot the max statistic


project_root = f"{Path.home()}/implied_motion"
target_decode = "fb"
design = "compact"
motion_type = "implied"

parser = argparse.ArgumentParser()
parser.add_argument("--workingdir", "-w")
parser.add_argument("--targetdecode", "-t")
parser.add_argument("--design", "-d")
parser.add_argument("--condition", "-c")

args = parser.parse_args()
project_root = args.workingdir
target_decode = args.targetdecode
design = args.design
motion_type = args.condition


def mean_confidence_interval(data, confidence=0.95):
    """mean_confidence_interval: A function that calculates the confidence
    interval (CI) based on the normal distribution using numpy and scipy.stats
    data: array like object
    confidence: float between 0 and 1
    returns: mean and CI of data"""
    m, se = np.mean(data), st.sem(data)
    h = se * st.norm.ppf((1 + confidence) / 2.0)
    return m, m - h, m + h


subjects = list(range(4, 24))
subjects.remove(19)
subjects = [f"sub-{str(x).zfill(2)}" for x in subjects]
plot_dir = f"{project_root}/data/permutation/plots"
early = ["V1", "V2", "V3"]

object = ["V3b", "hV4", "LO1", "LO2", "VO1", "VO2"]

motion = [
    "V3a",
    "hMT",
    "MST",
]

parietal = [
    "SPL1",
    "IPS_posterior",
    "IPS_anterior",
]


ROI_SETS = {"early": early, "object": object, "motion": motion, "parietal": parietal}

dfs = []
for set_name, roi_set in ROI_SETS.items():
    # SECOND STEP
    # empty list, that stores the full accuracy arrays per ROI, so that a CI can
    # be calculated
    roi_accuracies = []

    # empty lists where the mean accuracies and null distributions for ROIs will
    # be stored
    accuracy_mean_over_subs = []
    null_distribution_mean_over_subs = []
    for r, roi in enumerate(roi_set):
        # empty lists for accuracies and null distributions of subjects for the
        # current ROI
        accuracies = []
        null_distributions = []

        # create a figure. Here we plot all null distributions to make sure they
        # are symetrical and centured around chance level
        roi_fig = plt.figure()

        # inner loop - iterating over mask (=ROIs)
        for s, sub in enumerate(subjects):
            # read in the hdf5 data file for ROI[r] and SUBJECT[s]
            roi_file = f"{project_root}/data/decoding/{motion_type}/{sub}_{target_decode}_{roi}_{design}.csv"
            res = pd.read_csv(roi_file)

            # read out the accuracy and null distribution
            accuracies.append(res[res.type == "real"]["accuracy"].values[0])
            null_distributions.append(res[res.type == "permutation"]["accuracy"].values)

            # plot null distribution of subject for current ROI
            plt.hist(
                res[res.type == "permutation"]["accuracy"].values,
                bins=30,
                color="blue",
                alpha=1 / len(subjects),
            )

        # append the accuracies list to the roi_accuracies list to later plot their
        # confidence intervals
        roi_accuracies.append(accuracies)
        # calculate mean accuracy and null distribution of current ROI over
        # subjects. Append both to outer lists
        mean_accuracy = np.mean(accuracies)
        mean_null_distribution = np.mean(null_distributions, axis=0)
        accuracy_mean_over_subs.append(mean_accuracy)
        null_distribution_mean_over_subs.append(mean_null_distribution)

    # THIRD STEP
    # after getting all mean null distributions get max-null statistic
    null_distribution_mean_over_subs = np.asarray(null_distribution_mean_over_subs)
    max_statistic_null_distribution = null_distribution_mean_over_subs.max(axis=0)

    # calculate p-values for each ROI by getting the number of accuracies larger
    # than the 'real' accuracy
    ps = []
    for acc in accuracy_mean_over_subs:
        ps.append(sum(max_statistic_null_distribution > acc))

    ps = np.asarray(ps)
    ps = (ps + 1) / (max_statistic_null_distribution.shape[0] + 1)

    result_dict = {
        "ROIs": roi_set,
        "mean_accuracies": accuracy_mean_over_subs,
        "p_values": ps,
    }

    # FOURTH STEP
    # plot confidence intervalls for all ROIs and max statistic distribution in
    # one figure and draw significance threshold and theoretical chance level
    CI_fig, ax = plt.subplots(1, 2, sharey=True, figsize=(20, 10))
    ax[0].hist(
        max_statistic_null_distribution,
        bins=50,
        orientation="horizontal",
        label="max null distribution",
    )
    # draw the significance threshold (alpha=0.05 -> 95% percentile)
    ax[0].axhline(
        np.percentile(max_statistic_null_distribution, 95),
        color="green",
        label="alpha=0.05 threshold",
    )
    ax[1].axhline(
        np.percentile(max_statistic_null_distribution, 95),
        color="green",
        label="alpha=0.05 threshold",
    )

    # draw the chance level
    ax[0].axhline(1 / 2, color="red", linestyle="--", label="Chance level")
    ax[1].axhline(1 / 2, color="red", linestyle="--", label="Chance level")

    # empty lists for CI-bounds
    lows = []
    highs = []
    # for all ROIs plot the mean and CI
    for a, accs in enumerate(roi_accuracies):
        mean, low, high = mean_confidence_interval(accs)
        lows.append(low)
        highs.append(high)
        ax[1].plot([a, a], [low, high], "b")
        ax[1].plot(a, mean, "b*")

    # figure make up
    plt.xticks(np.arange(len(roi_set)), roi_set, rotation=45)
    ax[0].legend()
    ax[1].legend()
    CI_fig.savefig(
        f"{plot_dir}/{target_decode}_{motion_type}_{set_name}_{design}_accuracy_CI.png"
    )

    # save the mean accuracies for the ROIs, the max statistic null distribution
    # and the p-values for decoding accuracies for the ROIs
    result_dict["CI_lower"] = lows
    result_dict["CI_higher"] = highs
    result_dict["permutation_chance"] = np.percentile(
        max_statistic_null_distribution, 95
    )

    result_df = pd.DataFrame(data=result_dict, columns=result_dict.keys())
    result_df["target_decode"] = target_decode
    dfs.append(result_df)

mega_result_df = pd.concat(dfs)
mega_result_df.to_csv(
    f"{project_root}/data/permutation/{target_decode}_{motion_type}_{design}_max_statistics.csv",
    index=False,
)
