#!/bin/bash
#
# Wrapper script calling: make_datafiles.py (from https://github.com/abisee/cnn-dailymail)
#
# Data set URL: https://cs.nyu.edu/~kcho/DMQA/ -- files are in google docs,
# so we can not download via wget; must click through manually (or script it).
# (Data may be uploaded to s3 for convenience.)
#
# Downloaded data files (md5):
#   85ac23a1926a831e8f46a6b8eaf57263  cnn_stories.tgz
#   f9c5f565e8abe86c38bfa4ae8f96fd72  dailymail_stories.tgz
#
# Archive size                  |  Extracted size
#   151M  cnn_stories.tgz       |     557M  cnn/stories
#   358M  dailymail_stories.tgz |    1349M  dailymail/stories
#
# Resulting directories containing processed data:
#   ( 572M)  cnn_stories_tokenized/
#   (1374M)  dm_stories_tokenized/
#   (3022M)  finished_files/        <== this is what you want
#
# In s3, the pre-processed data is stored:
#   finished_files/
#        |-- test.bin.7z
#        |-- train.bin.7z
#        |-- val.bin.7z
#        \-- vocab.7z
#
##############################################################################

# $ python2 make_datafiles.py
# USAGE: python make_datafiles.py <cnn_stories_dir> <dailymail_stories_dir>

# Pass in "-n" as first option to just print out the command & not run it
[ "$1" = "-n" ] && RUN=echo || RUN=

die() { printf "** error: $@\n"; exit 2; }

# verify directory is clean
for dir in cnn_stories_tokenized dm_stories_tokenized finished_files; do
    [ -d "$dir" ] && print "** warning: output directory exists: $dir\n"
done

[ ! -d url_lists ] && die "missing input directory: url_lists"

# get standford-corenlp from https://stanfordnlp.github.io/CoreNLP/
# check if jar already exists before downloading/unzipping
#nlp_url='http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip' # newer, does not work
nlp_url='http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip'  # version used by abisee
nlp_dir=$(basename "$nlp_url" .zip)
nlp_zip="${nlp_dir}.zip"

nlp_jar=$(echo ${nlp_dir}/*stanford-corenlp-*[0-9].jar)
[ ! -f "$nlp_zip" -a ! -f "$nlp_jar" ] && { wget "$nlp_url"; unzip -q "$nlp_zip"; }

nlp_jar=$(echo ${nlp_dir}/*stanford-corenlp-*[0-9].jar)
[ -f "$nlp_jar" ] || die "NLP jar not found"

if [ ! -d cnn/stories -o ! -d dailymail/stories ]; then
    echo
    echo '# To copy CNN/DM data files from s3:'
    echo '#   aws s3 --profile w266 cp s3://michaeln-mids-data/cs.nyu.edu-kcho-DMQA/dailymail_stories.tgz .'
    echo '#   aws s3 --profile w266 cp s3://michaeln-mids-data/cs.nyu.edu-kcho-DMQA/cnn_stories.tgz .'
    echo '# or:'
    echo '#   aws s3 cp s3://michaeln-mids-data/cs.nyu.edu-kcho-DMQA/ .  --recursive --exclude "*" --include "*.tgz"'

    [ ! -f dailymail_stories.tgz ] &&
      aws s3 --profile w266 cp s3://michaeln-mids-data/cs.nyu.edu-kcho-DMQA/dailymail_stories.tgz .
    [ ! -f cnn_stories.tgz ] &&
      aws s3 --profile w266 cp s3://michaeln-mids-data/cs.nyu.edu-kcho-DMQA/cnn_stories.tgz .

    md5sum -c cnn-dm-md5.txt || die "md5 checksums for data files do not match"

    tar -xf cnn_stories.tgz
    tar -xf dailymail_stories.tgz
fi

printf "== cnn/stories:      \t %s  \t (expected:  92579)\n"   "$( ls -1 cnn/stories|wc -l )"
printf "== dailymail/stories:\t %s  \t (expected: 219506)\n\n" "$( ls -1 dailymail/stories|wc -l)"

## eg: export CLASSPATH=/path/to/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar
export CLASSPATH="${nlp_jar}:$CLASSPATH"
echo "Testing-Tokenizer CoreNLP-Enabled" | java edu.stanford.nlp.process.PTBTokenizer || die "tokenizer not found"
echo

$RUN python2 make_datafiles.py cnn/stories dailymail/stories

