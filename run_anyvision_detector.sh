#!/bin/bash

#eval "$(conda shell.bash hook)"
#conda activate MYENV


#  ------------- path args --------------------------------------------
# data_path: data path to video or dir
data_path=/home/nivpekar/projects/data/videos/outdoor
# bbs_video_dir: where to create the a videos with bboxes (if create_bbs_video==1)
bbs_video_dir=./
# bboxs_dir: where to create the files of the bboxs
bboxs_dir=./


# -----------------------------------------------------------------

python anyvision_detector.py --data_path $data_path \
--create_bbs_video 0 \
--bbs_video_dir $bbs_video_dir \
--bboxs_dir $bboxs_dir \
--two_patches 0 \
--NMSThreshold 0.1 \
--scoreThreshold 0.1

