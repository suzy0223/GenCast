#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "CUBLAS_WORKSPACE_CONFIG is set to $CUBLAS_WORKSPACE_CONFIG"

function evalcmd () {

    echo $1

    eval $1

    sleep 0.5s

}


dataset="metr_la"
split_type=(1 2 3 4)
epochs=100
quantile=1

for ((c=0; c<4; c++))
do
    wholecommand="python -u run_model.py --quantile ${quantile} --epochs ${epochs} --dataset ${dataset} --split_type ${split_type[$c]}"
    evalcmd "$wholecommand"
done