# YOLOX

### 가상 환경 설정

```shell
conda create -n yolox python=3.7

conda activate yolox

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

apt-get install gcc

apt-get install g++

pip3 install -v -e .
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
