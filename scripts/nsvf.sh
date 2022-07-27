#!/bin/bash

REPO_DIR=$(dirname ${0})
workspaceRoot="$( cd ${REPO_DIR} >/dev/null 2>&1 && pwd )"/..
cd ${workspaceRoot}

project=Mic

python3 train.py --root_dir data/Synthetic_NeRF/${project}/ --dataset_name nsvf --exp_name ${project} --num_epochs 3 --downsample 1.0 