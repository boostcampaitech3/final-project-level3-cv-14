#!/bin/bash
# bash run.sh 명령어를 통해서 실행할 수 있습니다!

####################### 이 부분만 수정하세요! #######################

DIR_NAME='[name0]name_of_experiments' # 템플릿 폴더 이름
BATCH_SIZE=64 # 배치 사이즈
SEED=42 # 시드 설정
EVAL_INTERVAL=10

# WANDB Setting
WANDB_LOG=true # true 값으로 설정해야 Wandb에 로그가 올라갑니다.
PROJECT_NAME='YOLOX_NANO_TRAIN_NOTA' # Wandb Project Name

################################ END ################################

if [ "$WANDB_LOG" = true ] ; then
    python tools/train.py -f yolox/models/$DIR_NAME/yolox_nano_exp.py -i $EVAL_INTERVAL -s $SEED -b $BATCH_SIZE --logger wandb wandb-project $PROJECT_NAME wandb-name $DIR_NAME
else
    python tools/train.py -f yolox/models/$DIR_NAME/yolox_nano_exp.py -i $EVAL_INTERVAL -s $SEED -b $BATCH_SIZE
fi