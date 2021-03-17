###
 # @FilePath: /yolov5/detect.sh
 # @Description:
 # @Author: zy
 # @Date: 2021-03-17 11:16:08
 # @LastEditTime: 2021-03-17 14:17:22
 # @LastEditors: zy
###

export LD_LIBRARY_PATH=/home/zy/anaconda3/lib:$LD_LIBRARY_PATH && export UMEXPR_MAX_THREADS=16

# - 官方 demo 数据
# python3 detect.py --source data/images --weights ../models/yolov5s.pt --conf 0.7 --device 0


# - 自定义数据
python3 detect.py --img-size 640 --conf 0.2 --device 0 \
  --source /home/zy/data_set/diy_dataset_yolov5/HD_winsize_2048_stepsize_512_threshold_0.7/images/test/ \
  --weights /home/zy/application/yolov5/yolov5/runs/train/exp12/weights/best.pt