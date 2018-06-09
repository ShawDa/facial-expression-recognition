# Facial Expression Recognition

Facial expression recognition using convolutional neural network.

## Overview
### Requirement
- Python3.5
- opencv
- Keras
- tensorflow-gpu
- tflearn

### Data
[Kaggle_fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data):Include 35587 lableled images, you can download `fer2013.tar.gz` and decompress `fer2013.csv` in the `data` folder.

[RAF_Dataset](http://www.whdeng.cn/RAF/model1.html):Include 15399 basic images and 3954 compound images.

Some processed data can be found here: https://pan.baidu.com/s/14xwd8YeTFk_LDKVn0YSbQA PWD: xxm5

### Howtouse
- get data
- run ```python3 data_process.py``` to generate npy files for training

### Results
I find some pics in [MS emotion_recognition API](https://azure.microsoft.com/zh-cn/services/cognitive-services/face/#recognition)

## Reference
https://github.com/isseu/emotion-recognition-neural-networks

## Citation
    @inproceedings{li2017reliable,
      title={Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild},
      author={Li, Shan and Deng, Weihong and Du, JunPing},
      booktitle={2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      pages={2584--2593},
      year={2017},
      organization={IEEE}
    }

