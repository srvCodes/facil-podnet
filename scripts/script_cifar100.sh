#!/bin/bash

if [ "$1" != "" ]; then
    echo "Running approach: $1"
else
    echo "No approach has been assigned."
fi
if [ "$2" != "" ]; then
    echo "Running on gpu: $2"
else
    echo "No gpu has been assigned."
fi

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd .. && pwd )"
SRC_DIR="$PROJECT_DIR/src"
echo "Project dir: $PROJECT_DIR"
echo "Sources dir: $SRC_DIR"

RESULTS_DIR="$PROJECT_DIR/results"
if [ "$4" != "" ]; then
    RESULTS_DIR=$4
else
    echo "No results dir is given. Default will be used."
fi
echo "Results dir: $RESULTS_DIR"

for SEED in 1
do
  if [ "$3" = "base" ]; then
          PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name base_${SEED} \
                 --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed $SEED \
                 --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR \
                 --gridsearch-tasks 10 --gridsearch-config gridsearch_config \
                 --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                 --approach $1 --gpu $2
  elif [ "$3" = "fixd" ]; then
          PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name fixd_${SEED} \
                 --datasets cifar100_icarl --num-tasks 6 --nc-first-task 50 --network resnet_rebuffi --seed $SEED \
                 --nepochs 2 --batch-size 128 --results-path $RESULTS_DIR \
                 --gridsearch-tasks 10 --gridsearch-config gridsearch_config \
                 --gridsearch-acc-drop-thr 0.2 --gridsearch-hparam-decay 0.5 \
                 --approach $1 --gpu $2 \
                 --num-exemplars 2000 --exemplar-selection herding --patience 15
  elif [ "$3" = "grow" ]; then
          PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name grow_${SEED} \
                 --datasets cifar100_icarl --num-tasks 51 --nc-first-task 50 --network resnet_rebuffi --seed $SEED \
		         --nepochs 160  --batch-size 128 --results-path $RESULTS_DIR \
                 --approach $1 --gpu $2 --lr 0.1 --lr-min 1e-5 --lr-factor 10 --momentum 0.9 \
		            --weight-decay 0.0005 --lr-patience 15 \
                 --num-exemplars-per-class 20 --exemplar-selection herding
  else
          echo "No scenario provided."
  fi
done
