#!/bin/bash

if (( $# != 2 )); then
    echo "Syntax $0 <frames_dir> <output_file>"
    exit
fi

frames_dir=$1
frames=($(ls -v $frames_dir))
output_file=$2

rm -f $output_file

for frame_file in ${frames[@]}; do
    detections_file=$frames_dir/$frame_file
    if [[ -s $detections_file ]]; then
        frame_id=${frame_file/.txt/}

        frame_data=($(cat $detections_file))

        # replace frame id
        frame_data[0]=$frame_id

        echo ${frame_data[@]} >> $output_file
    fi
done

