#!/usr/bin/env bash

rm -rf ./build
mkdir ./build
rsync -avz --no-links --exclude-from ~/mclnet/rsync_exclude.txt ~/mclnet ./build/
# rsync -avz --no-links /usr/local/cuda-10.0 ./
rsync -avz ../habitat-api/habitat/core/benchmark.py ./build/

rsync -avz   --exclude-from ./rsync_exclude_habitat.txt ../habitat-api ./build/
rsync -avz --exclude-from ./rsync_exclude_habitat.txt  ../habitat-sim ./build/


docker build . -t habitat_submission