###
 # @FilePath: /yolov5/train.sh
 # @Description:
 # @Author: zy
 # @Date: 2021-03-17 11:16:08
 # @LastEditTime: 2021-03-17 16:55:00
 # @LastEditors: zy
###

export LD_LIBRARY_PATH=/home/zy/anaconda3/lib:$LD_LIBRARY_PATH && export UMEXPR_MAX_THREADS=16

python3 train.py --img 640 --batch 32 --epochs 600   \
    --data /home/zy/application/yolov5/yolov5/zy_dir/data/data_cancer.yaml   \
    --weights /home/zy/application/yolov5/models/yolov5s.pt   \
    --hyp data/hyp.scratch.zy.yaml