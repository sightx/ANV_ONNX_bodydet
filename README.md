


to create the conda env run:

conda create --name ANV_DETECTOR --file requirements.txt

after that, activate the env:

conda activate ANV_DETECTOR


after that you can run the detector:

python anyvision_detector.py --data_path $data_path \
--create_bbs_video 0 \
--bbs_video_dir $bbs_video_dir \
--bboxs_dir $bboxs_dir \
--two_patches 0 \
--NMSThreshold 0.1 \
--scoreThreshold 0.1


create_bbs_video: 1 to create a video with the detections, 0 - dont create
bbs_video_dir: where to create the a videos with bboxes (if create_bbs_video==1)
bboxs_dir: where to create the files of the bboxs
two_patches: 1 for two patch, 0 for one patch





