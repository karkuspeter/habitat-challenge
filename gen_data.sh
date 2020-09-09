#!/usr/bin/env bash

DOCKER_NAME="habitat_submission"
GPU=0
PARAMS=""

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
      --gpu)
      shift
      GPU="${1}"
	  shift
      ;;
      --params)
      shift
      PARAMS="${@}"
      break  # exit the while loop
      ;;
    *) # unknown arg
      echo unkown arg ${1}. Arguments to be passed to agent.py should be added after --params
      exit
      ;;
esac
done

docker run -ti -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
    -v $(pwd)/temp:/temp \
    --runtime=nvidia \
    -e "AGENT_EVALUATION_TYPE=local" \
    -e "TRACK_CONFIG_FILE=/challenge_pointnav2020.local.rgbd.yaml" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_VISIBLE_DEVICES=${GPU} \
    ${DOCKER_NAME}\
    /bin/bash -c \
    "source activate habitat && ipython gen_data_agent.py --pdb -- ${PARAMS}; bash"
#     "export CONFIG_FILE=/gibson-challenge-data/locobot_p2p_nav_house.yaml$CONFIG_EXTENSION; export SIM2REAL_TRACK=$SIM2REAL_TRACK; cp /gibson-challenge-data/global_config.yaml$CONFIG_EXTENSION /opt/GibsonEnvV2/gibson2/global_config.yaml; ipython agent.py --pdb -- --gibson_mode evalsubmission --gibson_split evaltest; bash"


