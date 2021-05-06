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
    frame_id=${frame_file/.txt/}

    frame_data=($(cat $frames_dir/$frame_file))

    # replace frame id
    frame_data[0]=$frame_id

    echo ${frame_data[@]} >> $output_file
done

