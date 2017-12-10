#!/bin/bash
#
# Run this instead of run_make_datafiles.sh to download the data from s3.
#
# $ aws s3 --profile w266 ls s3://michaeln-mids-data/JafferWilson/
# 2017-11-15 04:32:03  207268941 cnn_stories_tokenized.zip
# 2017-11-15 04:34:28  482735659 dm_stories_tokenized.zip
# 2017-11-15 04:20:20 1004606087 finished_files.zip
# 2017-11-15 04:40:13       2425 readme.md
#
##############################################################################

# Pass in "-n" as first option to just print out the command & not run it
[ "$1" = "-n" ] && RUN=echo || RUN=

die() { printf "** error: $@\n"; exit 2; }

check_counts() {
    printf "== finished_files/chunked/val*   :\t %s \t (expected: 14)\n" "$(ls -1 finished_files/chunked/val* | wc -l)"
    printf "== finished_files/chunked/test*  :\t %s \t (expected: 288)\n" "$(ls -1 finished_files/chunked/test* | wc -l)"
    printf "== finished_files/chunked/train* :\t %s \t (expected: 12)\n" "$(ls -1 finished_files/chunked/train* | wc -l)"
    printf "== cnn_stories_tokenized :\t %s \t (expected: 92579)\n" "$(ls -1 cnn_stories_tokenized | wc -l)"
    printf "== dm_stories_tokenized  :\t %s \t (expected: 219506)\n" "$(ls -1 dm_stories_tokenized  | wc -l)"
}


profile='--profile w266'

[ ! -f readme.md ] && $RUN aws s3 $profile cp s3://michaeln-mids-data/JafferWilson/readme.md .

for f in cnn_stories_tokenized.zip \
         dm_stories_tokenized.zip \
         finished_files.zip
do
    [ ! -f "$f" ] && $RUN aws s3 $profile cp s3://michaeln-mids-data/JafferWilson/$f .
    dir=$(basename $f .zip)
    if [ -f $dir ]; then
        printf "** warning: output directory already exists, skipping: $dir\n"
    else
        printf "...create: $dir: "
        echo "...unzip: $f" && $RUN unzip -q "$f"
    fi
done

check_counts 2>/dev/null

