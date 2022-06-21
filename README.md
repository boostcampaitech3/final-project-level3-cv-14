## 프로젝트 개요
<div align="center">
<img src="https://user-images.githubusercontent.com/64190071/172797132-9fecb534-fbcd-49f4-99ca-1f52ce7537e5.png" width=30%>
</div>

인공지능 모델 압축 (경량화) 기술 기반 On-device AI를 추구하는 [Nota AI](https://www.nota.ai/)와 연계하여 진행한 YOLOX-nano 모델 경량화 프로젝트입니다.
### 프로젝트 기간
2022.05.16 ~ 2022.06.07

### 문제 정의
YOLOX-Nano 모델 구조 변경을 통한 YOLOX-nano 정확도(mAP) 개선


### 제약 사항
- 모델의 FLOPs와 Parameter 수는 기준 모델보다 작거나 같아야 함
- 고도화된 학습 기법을 이용한 성능 개선 금지
- Dataset : Pascal VOC 2007


### 수행 내용
모델을 구성하는 **레이어, 합성곱 블록 등을 변형**하여 베이스라인 모델의 parameter 개수와 GFLOPs를 초과하지 않으면서 **더욱 정확도(mAP) 높은 모델**을 생성

### 최종 결과

![image](https://user-images.githubusercontent.com/64190071/172392206-0e345df5-0ae1-47c4-af31-2471f9207ce4.png)

먼저, 기존의 모델에서 구조적 개선을 통한 경량화를 진행하여 **최대한 성능을 유지하면서 FLOPs와 Parameters를 줄일 수 있었습니다.**

여유 공간을 확보한 후, 성능 향상 기법을 적용하여 기존 모델 대비 **Parameters는 약 11.1% 감소**, mAP@.5:.95는 약 12.4% 상승이라는 결과를 얻어냈습니다.

|Model | Params<br>(M) |FLOPs<br>(G)|mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 |
| ------        |  :---:       |:---:     |:---:  | :---: |
|YOLOX-Nano Base |0.90  | 1.05 |28.29 | 50.64 |
|**Ours** |**0.80** | **1.05** | **31.81** | **53.75** |

#### 변경 내용
- 구조 개선을 통한 경량화
  - Backbone 구조 개선
  - FPN 구조 개선
- 성능 향상 기법
  - Mobile Bottleneck
  - GhostNet Convolution
  - Spatial Attention

Methods별 실험 결과

<img src = "https://user-images.githubusercontent.com/64190071/172396797-e08e0801-90b3-47a6-9530-e73ee0651fc8.png" width="60%">

## 사용 방법
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
  
  
### 팀원

|                                                  [김찬혁](https://github.com/Chanhook)                                                   |                                                                              [문태진](https://github.com/moontaijin)                                                  |                                                                        [이인서](https://github.com/Devlee247)                                                                         |                                                                         [장상원](https://github.com/agwmon)                                                                         |
| :-------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![KakaoTalk_20220421_103825891_03](https://user-images.githubusercontent.com/64190071/164358205-2048f3c2-1216-4836-a77f-a25de6a9091c.jpg) | ![KakaoTalk_20220421_103825891_04](https://user-images.githubusercontent.com/64190071/164358227-ef0d7919-bd0d-4a9d-8d50-42757a5c3534.jpg) | ![KakaoTalk_20220421_103825891_02](https://user-images.githubusercontent.com/64190071/164358185-a63371d7-84ad-4eb9-8337-c70857c0e170.jpg) | ![KakaoTalk_20220421_103825891_01](https://user-images.githubusercontent.com/64190071/164358129-a9ce91f8-84c5-4a9c-8329-27cf18e68e7f.jpg) |

  
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
