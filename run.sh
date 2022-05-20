#!/bin/bash

# bash run.sh 명령어를 통해서 실행할 수 있습니다!

####################### 이 부분만 수정하세요! #######################

# 복사한 템플릿 폴더 이름만 적으시면 됩니다.
DIR_NAME='[name0]name_of_experiments'
BATCH_SIZE=64
SEED=42

# WANDB Setting
WANDB_LOG=true
PROJECT_NAME='test'

####################### 윗 부분만 수정하세요! #######################

if [ "$WANDB_LOG" = true ] ; then
    python tools/train.py -f yolox/models/$DIR_NAME/yolox_nano_exp.py -s $SEED -b $BATCH_SIZE --logger wandb wandb-project $PROJECT_NAME wandb-name $DIR_NAME
else
    python tools/train.py -f yolox/models/$DIR_NAME/yolox_nano_exp.py -s $SEED -b $BATCH_SIZE
fi