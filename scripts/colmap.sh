#!/bin/bash

REPO_DIR=$(dirname ${0})
workspaceRoot="$( cd ${REPO_DIR} >/dev/null 2>&1 && pwd )"/..
cd ${workspaceRoot}

project=kitchen

python3 train.py --root_dir data/Colmap/${project}/ --num_epochs 4 --dataset_name colmap --exp_name ${project} --scale 16 --downsample 1.0 