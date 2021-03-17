## Reference

1. https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
2. https://github.com/PeterH0323/Smart_Construction
3. https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart
4. https://www.kaggle.com/c/global-wheat-detection/notebooks
5. https://www.kaggle.com/c/global-wheat-detection/discussion/172436
6. https://www.kaggle.com/nvnnghia/yolov5-pseudo-labeling
7. https://rwightman.github.io/pytorch-image-models/
8. [Yolov5 dataset insights to improve mAP](https://github.com/ultralytics/yolov5/issues/895)
9. https://github.com/lutzroeder/netron
10. [23 款神经网络的设计和可视化工具（8.12 更新）](https://zhuanlan.zhihu.com/p/147462170)
11. [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)


## 运行

**注意：**
1. 无特殊说明，默认情况下，以下命令在项目根目录下执行！

2. 打开新的 shell 终端时，要设置环境变量:
```bash
$ export LD_LIBRARY_PATH=/home/zy/anaconda3/lib:$LD_LIBRARY_PATH && export UMEXPR_MAX_THREADS=16
```

### 模型训练

- 训练日志可视化查看:
`$ tensorboard --logdir runs/train`

浏览器查看: `http://localhost:6006/`

- 官方 demo 数据训练

```bash
$ python3 train.py --img 640 --batch 8 --epochs 5 \
  --data coco128.yaml \
  --weights /home/zy/application/yolov5/models/yolov5s.pt
```

- 自定义数据的模型训练

```bash
$ python3 train.py --img 640 --batch 32 --epochs 1000   \
    --data /home/zy/application/yolov5/yolov5/zy_dir/data/data_cancer.yaml   \
    --weights /home/zy/application/yolov5/models/yolov5s.pt   \
    --hyp data/hyp.scratch.zy.yaml
```

**当前较好的模型**

1. `/home/zy/application/yolov5/yolov5/runs/train_exp30/weights`

来源:

```bash
python3 -u train.py --batch 32 --img 512 --epochs 1000 \
  --data /home/zy/application/yolov5/yolov5/zy_dir/data/data_cancer.yaml \
  --weights /home/zy/application/yolov5/models/yolov5m.pt \
  --hyp /home/zy/application/yolov5/yolov5/data/hyp.finetune.yaml
```

2. `runs/train/exp13/weights/last.pt`

来源:

```bash
$ python3 train.py --img 640 --batch 32 --epochs 1000   \
    --data /home/zy/application/yolov5/yolov5/zy_dir/data/data_cancer.yaml   \
    --weights /home/zy/application/yolov5/models/yolov5s.pt   \
    --hyp data/hyp.scratch.zy.yaml
```

训练参数设置:

```
试验: exp13

全量训练数据, 600 epochs

参数设置:

obj: 0.5  # obj loss gain (scale with pixels)

flipud: 0.5  # image flip up-down (probability)

mixup: 1.0  # image mixup (probability)


性能:

Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|██████████| 15/15 [00:10<00:00,  1.42it/s]
                 all         459         728       0.475       0.346       0.319      0.0919
        inflammation         459         319        0.48       0.464       0.403       0.132
              cancer         459         409       0.469       0.227       0.235      0.0514
Optimizer stripped from runs/train/exp13/weights/last.pt, 14.4MB
Optimizer stripped from runs/train/exp13/weights/best.pt, 14.4MB
Images sizes do not match. This will causes images to be display incorrectly in the UI.
600 epochs completed in 12.824 hours.
```


### 模型测试

模型测试: 在测试集上的表现

HD dataset

当前最好模型: **/home/zy/application/yolov5/yolov5/runs/train/exp/weights/best.pt**

```bash
$ python3 test.py --img-size 640 --batch-size 32 --conf-thres 0.3 --device 0 --task test --augment --save-json \
  --weights /home/zy/application/yolov5/yolov5/runs/train/exp/weights/last.pt \
  --data /home/zy/application/yolov5/yolov5/zy_dir/data/data_cancer.yaml
```


InflamCancer

```bash
$ python3 test.py --img-size 640 --batch-size 32 --conf-thres 0.3 --device 0 --task test --augment --save-json \
  --weights /home/zy/application/yolov5/yolov5/runs_InflamCancer/train/exp13/weights/last.pt \
  --data /home/zy/application/yolov5/yolov5/zy_dir/data/data_cancer.yaml
```

### 模型应用

- 官方 demo 数据

```bash
$ python3 detect.py --source data/images --weights ../models/yolov5s.pt --conf 0.7 --device 0
```

- 自定义数据

```bash
$ python3 detect.py --img-size 640 --conf 0.5 --device 0 \
  --source /home/zy/data_set/diy_dataset_yolov5/InflamCancer/images/test \
  --weights /home/zy/application/yolov5/yolov5/runs/train/exp13/weights/last.pt
```


## 其他

- 数据下载

1. VOCtrainval_06-Nov-2007.zip

`$ wget -c https://github.com/ultralytics/yolov5/releases/download/v1.0/VOCtrainval_06-Nov-2007.zip`

2. 样例数据下载:

`$ wget -c https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip`

3. yolov5 镜像下载:

`$ sudo docker pull ultralytics/yolov5:latest`


- glibc 版本问题

```
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by /home/zy/anaconda3/lib/python3.7/site-packages/google/protobuf/pyext/_message.cpython-37m-x86_64-linux-gnu.so)

$ strings /home/zy/anaconda3/lib/libstdc++.so.6 | grep GLIBC

$ strings /usr/lib64/libstdc++.so.6 | grep GLIBC

发现 /home/zy/anaconda3/lib/libstdc++.so.6 满足版本要求，运行程序时，可以临时设置环境变量
$ export LD_LIBRARY_PATH=/home/zy/anaconda3/lib:$LD_LIBRARY_PATH

现在程序正常运行!
```


- yolov5 依赖的 pytorch

```
yolov5 依赖的 pytorch: torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2

Using torch 1.7.1+cu101 CUDA:0 (GeForce GTX 1080, 8119.1875MB)

pytorch 下载:
$ pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

分别下载到本地，然后 pip 本地安装
(1) torch==1.7.1+cu101
$ python3 -m pip download torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Looking in links: https://download.pytorch.org/whl/torch_stable.html
Collecting torch==1.7.1+cu101
  Downloading https://download.pytorch.org/whl/cu101/torch-1.7.1%2Bcu101-cp37-cp37m-linux_x86_64.whl (735.4 MB)

或
$ wget -c https://download.pytorch.org/whl/cu101/torch-1.7.1%2Bcu101-cp37-cp37m-linux_x86_64.whl

(2) torchvision==0.8.2+cu101

$ python3 -m pip download torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html

Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Looking in links: https://download.pytorch.org/whl/torch_stable.html
Collecting torchvision==0.8.2+cu101
  Downloading https://download.pytorch.org/whl/cu101/torchvision-0.8.2%2Bcu101-cp37-cp37m-linux_x86_64.whl (12.8 MB)

或
$ wget -c https://download.pytorch.org/whl/cu101/torchvision-0.8.2%2Bcu101-cp37-cp37m-linux_x86_64.whl

(3) torchaudio==0.7.2

$ python3 -m pip download torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Looking in links: https://download.pytorch.org/whl/torch_stable.html
Collecting torchaudio==0.7.2
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/37/16/ecdb9eb09ec6b8133d6c9536ea9e49cd13c9b5873c8488b8b765a39028da/torchaudio-0.7.2-cp37-cp37m-manylinux1_x86_64.whl (7.6 MB)

或
$ wget -c https://pypi.tuna.tsinghua.edu.cn/packages/37/16/ecdb9eb09ec6b8133d6c9536ea9e49cd13c9b5873c8488b8b765a39028da/torchaudio-0.7.2-cp37-cp37m-manylinux1_x86_64.whl

```

- [Netron](https://github.com/lutzroeder/netron)

```
Netron is a viewer for neural network, deep learning and machine learning models.

Netron supports ONNX (.onnx, .pb, .pbtxt), Keras (.h5, .keras), TensorFlow Lite (.tflite),
Caffe (.caffemodel, .prototxt), Darknet (.cfg), Core ML (.mlmodel), MNN (.mnn),
MXNet (.model, -symbol.json), ncnn (.param), PaddlePaddle (.zip, __model__),
Caffe2 (predict_net.pb), Barracuda (.nn), Tengine (.tmfile), TNN (.tnnproto),
RKNN (.rknn), MindSpore Lite (.ms), UFF (.uff).


Netron has experimental support for TensorFlow (.pb, .meta, .pbtxt, .ckpt, .index),
PyTorch (.pt, .pth), TorchScript (.pt, .pth), OpenVINO (.xml), Torch (.t7),
Arm NN (.armnn), BigDL (.bigdl, .model), Chainer (.npz, .h5), CNTK (.model, .cntk),
Deeplearning4j (.zip), MediaPipe (.pbtxt), ML.NET (.zip), scikit-learn (.pkl), TensorFlow.js (model.json, .pb).
```


## Issues

1. [How about training my data in pre_trained model and use something about it's mAP](https://github.com/ultralytics/yolov5/issues/2025)

2. [How to improve recall rate on specific class?](https://github.com/ultralytics/yolov5/issues/1995)

3. [How to improve the value of map0.5 and jump out of local optimum?](https://github.com/ultralytics/yolov5/issues/1985)
```
You may want to examine your individual loss components, and if one is overfitting
 more than the others, you might reduce it's contribution to the loss function by
 reducing it's associated hyperparameter, i.e. hyp['obj'] if objectness is overfitting most etc.


improving results is very simple:

Use a larger dataset
Use a larger model i.e. --weights yolov5x.pt
Use a larger image size i.e. --img 1280
Train longer i.e. --epochs 500
Evolve hyperparameters (see Tutorials) https://github.com/ultralytics/yolov5#tutorials
```

4. [Overfit with small dataset, like KITTI](https://github.com/ultralytics/yolov5/issues/821)

5. [The confidence or category probability value is too small](https://github.com/ultralytics/yolov5/issues/837)

6. [Object loss in yolov3 vs yolov5](https://github.com/ultralytics/yolov5/issues/1808)

7. [I don't know why val obj_loss, val obj_loss 很快就过拟合了：升高](https://github.com/ultralytics/yolov5/issues/1999)

8. [I don't know why val obj_loss↑](https://github.com/ultralytics/yolov5/issues/1932)