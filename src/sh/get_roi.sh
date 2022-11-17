#!/bin/bash


PROJECT_ROOT=$HOME/implied_motion

DOCKER_FREESURFER_ROOT=/home/data/derivatives/fMRIprep/sourcedata/freesurfer

declare -A ROIS=(["01"]="V1v" \
	["02"]="V1d" \
	["03"]="V2v" \
	["04"]="V2d" \
	["05"]="V3v" \
	["06"]="V3d" \
	["07"]="hV4" \
	["08"]="VO1" \
	["09"]="VO2" \
	["10"]="PHC1" \
	["11"]="PHC2" \
	["12"]="MST" \
	["13"]="hMT" \
	["14"]="LO2" \
	["15"]="LO1" \
	["16"]="V3b" \
	["17"]="V3a" \
	["18"]="IPS0" \
	["19"]="IPS1" \
	["20"]="IPS2" \
	["21"]="IPS3" \
	["22"]="IPS4" \
	["23"]="IPS5" \
	["24"]="SPL1" \
	["25"]="FEF")

# define freesurfer environment variables
FS_LICENSE=$DOCKER_FREESURFER_ROOT/license.txt 
FREESURFER_HOME=/usr/local/freesurfer/bin/freesurfer


for participant in 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20 21 22 23
do

	# get subject folder name
	SUB_FOLDER=sub-$(printf %02d "$participant")

	# name the fs container so u can exec commands on it
	FS_CONTAINER_NAME=agbartels_freesurfer_can_$SUB_FOLDER
		
	# define data related variables
	THIS_SUBJECT_DIR=$DOCKER_FREESURFER_ROOT/$SUB_FOLDER
	THIS_SUBJECT_MEAN=("$PROJECT_ROOT"/data/derivatives/fMRIprep/"$SUB_FOLDER"/func/"$SUB_FOLDER"_task*1_space-T1w_boldref.nii.gz)
	THIS_SUBJECT_MEAN_FILE="${THIS_SUBJECT_MEAN##*/}"
	THIS_SUBJECT_MEAN_DOCKER=/home/data/derivatives/fMRIprep/$SUB_FOLDER/func/$THIS_SUBJECT_MEAN_FILE

	# make ROI directory 
	PAR_ROI_FOLDER="$PROJECT_ROOT"/data/derivatives/wang_2015/"$SUB_FOLDER"

	if [ ! -d "$PAR_ROI_FOLDER" ]; then
		mkdir -p "$PAR_ROI_FOLDER"
	fi

	# generate surface labels from Wang et al. 2015
	if [[ "$participant" == 7 ]]
	then
		python -m neuropythy atlas --atlases wang15 --verbose "$SUB_FOLDER"
	fi

	docker run \
	--detach \
	--name "$FS_CONTAINER_NAME" \
	--interactive \
	--tty \
	--rm \
	--user "$(id -u):$(id -g)" \
	--volume "${PROJECT_ROOT}:/home" \
	--env FS_LICENSE=$FS_LICENSE \
	--env SUBJECTS_DIR=$DOCKER_FREESURFER_ROOT \
	--env FREESURFER_HOME=$FREESURFER_HOME \
	--cpus=2 \
	freesurfer/freesurfer:7.3.0

	# register mean epi to surface

	docker exec "$FS_CONTAINER_NAME" \
	tkregister2_cmdl --mov "$THIS_SUBJECT_MEAN_DOCKER" \
	--s "$SUB_FOLDER" --regheader --reg $DOCKER_FREESURFER_ROOT/"$SUB_FOLDER"/register_"$SUB_FOLDER".dat


	for hemisphere in lh rh
	do
		for roi_no in "${!ROIS[@]}"
		do

			echo running participant "$participant" roi "${ROIS[$roi_no]}" "$hemisphere"
		# get labels for the ROIs
			docker exec "$FS_CONTAINER_NAME" \
			mri_vol2label --i "$THIS_SUBJECT_DIR"/surf/$hemisphere.wang15_mplbl.mgz \
			--id "$roi_no" --l "$THIS_SUBJECT_DIR"/label/$hemisphere.wang2015atlas."${ROIS[$roi_no]}".label \
			--surf-path "$THIS_SUBJECT_DIR"/surf/$hemisphere.inflated

		#  create roi masks
			docker exec "$FS_CONTAINER_NAME" \
			mri_label2vol --label "$THIS_SUBJECT_DIR"/label/$hemisphere.wang2015atlas."${ROIS[$roi_no]}".label  \
			--temp "$THIS_SUBJECT_MEAN_DOCKER" \
			--reg $DOCKER_FREESURFER_ROOT/"$SUB_FOLDER"/register_"$SUB_FOLDER".dat \
			--fillthresh 0 --proj frac 0 1 .1 \
			--subject "$SUB_FOLDER" --hemi $hemisphere \
			--o /home/data/derivatives/wang_2015/"$SUB_FOLDER"/$hemisphere-"${ROIS[$roi_no]}"-roi.nii
		done
	done
	docker stop "$FS_CONTAINER_NAME"
done