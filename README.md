# YOLOX

### 가상 환경 설정

```shell
conda create -n yolox python=3.7

conda activate yolox

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

apt-get install gcc

apt-get install g++

pip3 install -v -e .

pip3 install wandb
```

### 데이터 생성

```shell
cd YOLOX/datasets
mkdir trainval
mkdir test

# train dataset
wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtrainval_06-Nov-2007.tar -C ./trainval
rm -rf VOCtrainval_06-Nov-2007.tar

# val dataset
wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar -C ./test
rm -rf VOCtest_06-Nov-2007.tar
```

### Train


#### Shell script
```shell
bash run.sh
```
위의 명령어로 바로 train을 진행할 수 있습니다.

shell script의 상세 구성은 다음과 같습니다.
```
####################### 이 부분만 수정하세요! #######################

DIR_NAME='[name0]name_of_experiments' # 템플릿 폴더 이름
BATCH_SIZE=64 # 배치 사이즈
SEED=42 # 시드 설정
EVAL_INTERVAL=10

# WANDB Setting
WANDB_LOG=true # true 값으로 설정해야 Wandb에 로그가 올라갑니다.
PROJECT_NAME='YOLOX_NANO_TRAIN_NOTA' # Wandb Project Name

################################ END ################################
```
`DIR_NAME`, `BATCH_SIZE`, `SEED`, `EVAL_INTERVAL`, `WANDB_LOG`, `PROJECT_NAME`을 수정하여 학습을 진행할 수 있습니다.

이 외에 블록 교체 등의 작업은 yolox/models/{DIR_NAME}에서

`yolox.py`, `yolo_pafpn.py`, `yolo_head.py`, `darknet.py`, `yolox_nano_exp.py`를 수정하여 진행하시면 됩니다.

이외에 Conv layer 등을 수정하려면 yolox/models/network_blocks.py를 수정하시면 됩니다.


#### by python
Shell script 말고도 학습을 진행할 수 있습니다.

```python
python tools/train.py -f exps/default/yolox_voc_nano.py -b 16 --logger wandb wandb-project <project name>
```

## Cite YOLOX
If you use YOLOX in your research, please cite our work by using the following BibTeX entry:

```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
