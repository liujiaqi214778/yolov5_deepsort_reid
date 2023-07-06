export PYTHONPATH=$PWD
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python person_search_reid.py --video_path ./ch01_20171125113450.mp4 \
 --weights ./weights/yolov5x6.pt --conf-thres 0.4 \
 --config-file ./configs/resnet-bot50-ibn-ma-ms-cs.yml \
 --opts MODEL.WEIGHTS ./weights/model_best.pth


# python person_search_reid.py --video_path ./ch01_20171125113450.mp4 --weights ./weights/yolov5x6.pt --config-file ./configs/resnet-bot50-ibn-ma-ms-cs.yml  --opts MODEL.WEIGHTS ./weights/model_best.pth