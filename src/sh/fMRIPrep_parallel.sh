#!/bin/bash

# submitting fmriprep for all participants to the server
# processing will be done in parallel

PROJECT_ROOT=$HOME/implied_motion


for par in 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20 21 22 23
do
    SUB_FOLDER=sub-$(printf %02d $par)
    qsub -F "--participant '$SUB_FOLDER'" \
    "$PROJECT_ROOT"/src/sh/fMRIPrep.pbs 
done
