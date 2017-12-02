#!/bin/bash -x

[ ! -f util/create_data.py ] && { echo "** error: run $0 from code directory."; exit 2; }

# first arg is output directory, e.g., ../output
default_args=
if [ $# -gt 0 ]; then
    base_dir=$1
    default_args="--out_dir $base_dir"
    mkdir "$base_dir"
    shift
fi

time python3 util/create_data.py $default_args $@

