#!/usr/bin/env bash

DOCKER_NAME="habitat_submission"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done

docker run -ti -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
    -v $(pwd)/temp:/temp \
    --runtime=nvidia \
    -e "AGENT_EVALUATION_TYPE=local" \
    -e "TRACK_CONFIG_FILE=/challenge_pointnav2020.local.rgbd.yaml" \
    ${DOCKER_NAME}\
    /bin/bash -c \
    "source activate habitat && bash submission.sh; bash"
#     "export CONFIG_FILE=/gibson-challenge-data/locobot_p2p_nav_house.yaml$CONFIG_EXTENSION; export SIM2REAL_TRACK=$SIM2REAL_TRACK; cp /gibson-challenge-data/global_config.yaml$CONFIG_EXTENSION /opt/GibsonEnvV2/gibson2/global_config.yaml; ipython agent.py --pdb -- --gibson_mode evalsubmission --gibson_split evaltest; bash"


