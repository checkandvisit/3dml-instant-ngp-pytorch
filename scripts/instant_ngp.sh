#!/bin/bash

REPO_DIR=$(dirname ${0})
workspaceRoot="$( cd ${REPO_DIR} >/dev/null 2>&1 && pwd )"/..
cd ${workspaceRoot}

project=kitchen

python3 train.py --root_dir data/InstantNGP/${project}/ --num_epochs 3 --dataset_name instant_ngp --exp_name ${project} --downsample 1.0 