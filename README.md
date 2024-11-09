  # EDSR_paddle

Paddle 复现版本

## 数据集

分类之后训练集用于训练SR模块
https://aistudio.baidu.com/aistudio/datasetdetail/106261
## aistudio
脚本任务地址: https://aistudio.baidu.com/aistudio/clusterprojectdetail/2356381
## 训练模型
Weigths/EDSR
## 训练步骤
### train sr
```bash
python train.py -opt config/train/train_EDSR_x2.yml
python train.py -opt config/train/train_EDSR_x3.yml
python train.py -opt config/train/train_EDSR_x4.yml
```
## 测试步骤
```bash
python test.py -opt config/test/test_EDSR_x2.yml
python test.py -opt config/test/test_EDSR_x3.yml
python test.py -opt config/test/test_EDSR_x4.yml
```

  # RRDBNet_paddle

Paddle 复现版本

## 数据集

分类之后训练集用于训练SR模块
https://aistudio.baidu.com/aistudio/datasetdetail/106261
## aistudio
脚本任务地址: https://aistudio.baidu.com/aistudio/clusterprojectdetail/2356381
## 训练模型
Weigths/RRDBNet
## 训练步骤
### train sr
```bash
python train.py -opt config/train/train_RRDBNet_x2.yml
```
## 测试步骤
```bash
python test.py -opt config/test/test_RRDBNet_x2.yml
```