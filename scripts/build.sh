#!/bin/bash

sudo pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
sudo pip3 install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

cd ${DEPENDENCIES}/apex
sudo pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

REPO_DIR=$(dirname ${0})
workspaceRoot="$( cd ${REPO_DIR} >/dev/null 2>&1 && pwd )"/..
cd ${workspaceRoot}

sudo pip3 install -r requirements.txt
sudo pip3 install models/csrc/
