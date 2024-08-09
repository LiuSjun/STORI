# STORI
Code and Description of STORI model

![STORI](Flowchat.png)

- We present a novel  Spatio-Temporal Optical Reflectance Interpolation (STORI) method, which integrates Synthetic Aperture Radar (SAR) data and optical reflectance data to reconstruct continuous, high-density reflectance time series. The algorithm does not require any prior knowledge about the spatial distribution.

- STORI effectively fuses SAR and optical data, leveraging the complementary strengths of these modalities. SAR data, with its all-weather capability, fills the gaps in optical data, enabling the generation of cloud-free time series with high temporal resolution.

- The lightweight design of our model allows for efficient and accurate reconstruction without the need for additional training data, making it broadly applicable across various geographical and climatic regions..

**This code is ONLY released for academic use.**

## How to use
We split STORI into two components:

- rank-reid
  - Framework: Pytorch
  - Training Resnet based Bi-LSTM network on source dataset
  - Learning to rank on target dataset
- TrackViz
  - Dependencies: Some traditional libraries, including numpy, pickle, matplotlib, seaborn
  - Building spatial temporal model with visual classification results
  - STORI Fusion

Written and tested in python2, Pytorch 1.12.4.

>Attention: make sure you are using the repos specified in STORI. You are possible to meet some errors if you use other version repos.

### Dataset
#### Download
 - [CIA]([https://data.csiro.au/collection/csiro:5846](https://data.csiro.au/collection/csiro:5846))
 - [LGC]([https://data.csiro.au/collection/csiro:5847v3](https://data.csiro.au/collection/csiro:5847v3))
 - [CHINA_LANDSAT_MODIS]([https://pan.baidu.com/s/1ymgud6tnY6XB5CTCXPUfnw](https://pan.baidu.com/s/1ymgud6tnY6XB5CTCXPUfnw))
 - [WorldView]([http://sshy3s.com/newsitem/27839360](http://sshy3s.com/newsitem/278393600))


#### Configuration
- Pretrain Config: Modify all path containing '/home/cwh' appearing in rank-reid/pretrain/pair_train.py  to your corresponding path.
- Fusion Config 
  - Modify all path containing '/home/cwh' appearing in TrackViz/ctrl/transfer.py  to your corresponding path.
  - Modify all path containing '/home/cwh' appearing in rank-reid/rank-reid.py  to your corresponding path.

### Pretrain
Pretrain Resnet52 and Siamese Network using source datasets.

```bash
cd rank-reid/pretrain && python pair_train.py
```

This code will save pretrained model in pair-train directory:

```bash
pretrain
├── cuhk_pair_pretrain.h5
├── cuhk_softmax_pretrain.h5
├── eval.py
├── grid-cv-0_pair_pretrain.h5
├── grid-cv-0_softmax_pretrain.h5
├── grid-cv-1_pair_pretrain.h5
├── grid-cv-1_softmax_pretrain.h5
├── grid-cv-2_pair_pretrain.h5
├── grid-cv-2_softmax_pretrain.h5
├── grid-cv-3_pair_pretrain.h5
├── grid-cv-3_softmax_pretrain.h5
├── grid-cv-4_pair_pretrain.h5
├── grid-cv-4_softmax_pretrain.h5
├── grid-cv-5_pair_pretrain.h5
├── grid-cv-5_softmax_pretrain.h5
├── grid-cv-6_pair_pretrain.h5
├── grid-cv-6_softmax_pretrain.h5
├── grid-cv-7_pair_pretrain.h5
├── grid-cv-7_softmax_pretrain.h5
├── grid-cv-8_pair_pretrain.h5
├── grid-cv-8_softmax_pretrain.h5
├── grid-cv-9_pair_pretrain.h5
├── grid-cv-9_softmax_pretrain.h5
├── grid_pair_pretrain.h5
├── grid_softmax_pretrain.h5
├── __init__.py
├── market_pair_pretrain.h5
├── market_softmax_pretrain.h5
├── pair_train.py
├── pair_transfer.py
├── source_pair_pretrain.h5
└── source_softmax_pretrain.h5

```

## TFusion
include directly vision transfering, fusion, learning to rank

```bash
cd TrackViz && python ctrl/transfer.py
```

Results will be saved in TrackViz/data

```bash
TrackViz/data
├── source_target-r-test # transfer after learning to rank on test set
│   ├── cross_filter_pid.log
│   ├── cross_filter_score.log
│   ├── renew_ac.log
│   ├── renew_pid.log
│   └── sorted_deltas.pickle
├── source_target-r-train # transfer after learning to rank on training set
│   ├── cross_filter_pid.log
│   ├── cross_filter_score.log
│   ├── cross_mid_score.log
│   ├── renew_ac.log
│   ├── renew_pid.log
│   └── sorted_deltas.pickle
├── source_target-r-train_diff # ST model built by random classifier minus visual classfier after learning to rank
│   ├── renew_pid.log
│   └── sorted_deltas.pickle
├── source_target-r-train_rand  # ST model built by random classifier after learning to rank
│   ├── renew_pid.log
│   └── sorted_deltas.pickle
├── source_target-test # directly transfer from source to target test set
│   ├── cross_filter_pid_32.log
│   ├── cross_filter_pid.log
│   ├── cross_filter_score.log
│   ├── renew_ac.log
│   ├── renew_pid.log
│   └── sorted_deltas.pickle
├── source_target-train # directly transfer from source to  target training set
│   ├── cross_filter_pid.log # sorted pids by fusion scores
│   ├── cross_filter_score.log # sorted fusion scores corresponding to pids
│   ├── cross_mid_score.log # can be use to generate pseudo lable, ignore it 
│   ├── renew_ac.log #  sorted vision scores corresponding to pids
│   ├── renew_pid.log # sorted pids by vision scores
│   └── sorted_deltas.pickle # store time deltas, so called ST model built by visual classifier
├── source_target-train_diff # store time deltas, ST model built by random classifier minus visual classifier
│   ├── renew_pid.log
│   └── sorted_deltas.pickle
└── source_target-train_rand # store time deltas, built by random visual classifier
    ├── renew_pid.log
    └── sorted_deltas.pickle
```

### Evaluation
Evaluation result will be automatically saved in the log_path, as you specified in rank-reid/rank-reid.py predict_eval(), default location is TrackViz/market_result_eval.log, TrackViz/grid_eval.log  

- GRID evaluation includes rank1, rank5, rank-10 accuracy
- Market-1501 evaluation includes rank1 accuracy and mAP. Rank5 and rank10 should be computed by code in [MATLAB](http://pan.baidu.com/s/1hqMbd4K) provided by Liang Zheng.
