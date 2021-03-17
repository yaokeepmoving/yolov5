###
 # @FilePath: /yolov5/test.sh
 # @Description:
 # @Author: zy
 # @Date: 2021-03-17 11:16:08
 # @LastEditTime: 2021-03-17 13:40:42
 # @LastEditors: zy
###

export LD_LIBRARY_PATH=/home/zy/anaconda3/lib:$LD_LIBRARY_PATH && export UMEXPR_MAX_THREADS=16

python3 test.py --img-size 640 --batch-size 32 --conf-thres 0.05 --device 0 --task test --augment --save-json \
  --weights /home/zy/application/yolov5/yolov5/runs/train/exp11/weights/best.pt \
  --data /home/zy/application/yolov5/yolov5/zy_dir/data/data_cancer.yaml