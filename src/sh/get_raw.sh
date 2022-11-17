#!/bin/bash

SOURCE_ROOT=/gpfs01/bartels/group/implied_motion/Experiments_fMRI/01_IMD/data/e01s
TARGET_ROOT=$HOME/implied_motion/data
TARGET_RAW=$TARGET_ROOT/raw
TARGET_LOG=$TARGET_ROOT/behavioural




for i in 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20 21 22 23
do
  SUB_FOLDER=$(printf %02d "$i")
  SOURCE_PAR=$SOURCE_ROOT$SUB_FOLDER
  
  TARGET_PAR_RAW=$TARGET_RAW/sub-$SUB_FOLDER
  TARGET_PAR_LOG=$TARGET_LOG/sub-$SUB_FOLDER
  TARGET_PAR_ANAT=$TARGET_PAR_RAW/structural



  SOURCE_PAR_FUNC=$SOURCE_PAR/session_exp/raw/.
  SOURCE_PAR_ANAT=$SOURCE_PAR/session_loc/structural/.
  SOURCE_PAR_LOG=$SOURCE_PAR/session_exp/logs/.



  rsync -r "$SOURCE_PAR_FUNC" "$TARGET_PAR_RAW"
  rsync -r "$SOURCE_PAR_ANAT" "$TARGET_PAR_ANAT"
  rsync -r "$SOURCE_PAR_LOG" "$TARGET_PAR_LOG"

  echo "$i" out of "$END_PAR" participant data moved.


done