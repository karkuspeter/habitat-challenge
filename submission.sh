#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=

python agent.py --habitat_eval $AGENT_EVALUATION_TYPE $@

