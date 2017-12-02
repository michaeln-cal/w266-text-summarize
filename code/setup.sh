#!/bin/bash -x

[ ! -f util/create_data.py ] && { echo "** error: run $0 from code directory."; exit 2; }

output=../output
#mkdir -p $output/abstract $output/article $output/result $output/vocab

time python3 util/create_data.py

