#!/bin/bash

PROJECT_ROOT=$HOME/implied_motion
DATA_RAW=$PROJECT_ROOT/data/raw
DATA_NIFTI=$PROJECT_ROOT/data/bids


for i in 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20 21 22 23
do

  SUB_FOLDER=sub-$(printf %02d $i)
  DATA_RAW_PAR=$DATA_RAW/$SUB_FOLDER
  DATA_NIFTI_PAR=$DATA_NIFTI/$SUB_FOLDER

  mkdir "$DATA_NIFTI_PAR"

  # explanation of flags
  # -o: output directory
  # -b: create bids sidecar (y or n)
  # -ba: anonymise bids sidecar (y or n)
  # last positional: input directory

  dcm2niix -o "$DATA_NIFTI_PAR" -b y -ba y "$DATA_RAW_PAR"

done