#!/bin/bash

# decode direction of motion using classifiers trained on motion of the same type
# loop through the following parameters and subject them as jobs to the head node
# loop over:
# participants
# rois
# design matrices (compact or expanded, used to obtain the beta estimates)
# decoding questions (lr -> left vs right, fb -> forward backward)
# motion type (real or implied)

PROJECT_ROOT=$HOME/implied_motion

for participant in 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20 21 22 23
do
    for roi in hMT hV4 IPS_anterior IPS_posterior LO1 LO2 MST V1 V2 V3 V3a V3b VO1 VO2 SPL1
    do
        for design in compact expanded
        do
            for target in lr fb
            do
                    qsub \
                    -l nodes=1:ppn=12 \
                    -l mem=4gb \
                    -l walltime=00:00:30:00 \
                    -F "--workingdir '$PROJECT_ROOT' --sub '$participant' --roi '$roi' --targetdecode '$target' --design '$design' --permutations 1000" \
                    "$PROJECT_ROOT"/src/Py/mvpa_cross_script.py              
            done
        done
    done
done
