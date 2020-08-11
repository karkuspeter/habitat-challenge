#!/usr/bin/env bash

#rm -rf ./build
#mkdir ./build
rsync -avz --no-perms --no-owner --no-group --no-links --delete --exclude-from ~/mclnet/rsync_exclude.txt ~/mclnet ./build/
# rsync -avz --no-links /usr/local/cuda-10.0 ./
rsync -avz --no-perms --no-owner --no-group  --delete ../habitat-api/habitat/core/benchmark.py ./build/

rsync -avz --no-perms --no-owner --no-group  --delete --exclude-from ./rsync_exclude_habitat.txt ../habitat-api ./build/
rsync -avz --no-perms --no-owner --no-group --delete --exclude-from ./rsync_exclude_habitat.txt  ../habitat-sim ./build/


docker build . -t habitat_submission