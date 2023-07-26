2023/6/1 liujiaqi
1. fastreid 换成新版
2. fastreid 模型改成 configs/resnet...yml 训练，旧框架的有问题
3. arg paser 传参有问题，暂时先直接把参数写在代码里，直接python person_search_reid.py启动
4. 模型 
  ./weights/yolov5s.pt
  ./weights/model_best.pth （2里面训练的）
1. 新fastreid模型输出2048维，query的图像特征是512，计算余弦相似度的代码会报错，明天我手动弄几个2048的query
2. mp4文件放在项目根目录./ch01_20171125113450.mp4

2023/6/6 liujiaqi
1. 手动截取的query.npy在 qilei.seu-palm.com -p 22  /data/ljq/reid-system 下对应目录，每100帧如果有人则取第1个框的feature加入query，总共大概1000多个query，代码已被注释
2. 把画上框和id的帧输出到test.jpg，会覆盖


2023/06/19 liujiaqi
1. gallery.py 动态维护行人特征库
2. 有行人的帧输出在pictures目录
3. video_writer.py 将一个目录的jpg转成mp4


2023/06/20 liujiaqi
1. 修复argparser的问题


2023/06/26 liujiaqi
1. 改用yolov5x.pt, 还是会有识别影子和部分躯干的问题，要重点解决
2. ReID检索放到deepsort中，以deepsort的输出画框。
3. 现在ReID+DeepSort可以保证一个帧的所有框不可能出现相同pid
4. ReID在gallery中还是只对每个pid存储一个静态特征向量（deepsort中person的第一个框）


2023/06/27 liujiaqi
1. 改用新yolov5代码, yolov5x6.pt
2. YOLO框和DeepSort框同时输出，调试用
3. Deepsort n_init -> 16, 框匹配16帧才打deepsort框 + reid search

# Getting Started

```bash
python person_search_reid.py --video_path ./ch01_20171125113450.mp4 \
--weights ./weights/yolov5x6.pt --conf-thres 0.4 \
--config-file ./configs/resnet-bot50-ibn-ma-ms-cs.yml \
--opts MODEL.WEIGHTS ./weights/model_best.pth
```

# 网络流视频处理+转发
需要摄像头支持rtsp、rtmp等协议
--video_path 指定摄像头或网络流地址
--ffmpeg 指定推流的rtsp服务器地址

```bash
python person_search_reid.py --video_path 'rtsp://admin:*******@x.x.x.x/Streaming/Channels/1' \
--weights ./weights/yolov5x6.pt --conf-thres 0.4 \
--config-file ./configs/resnet-bot50-ibn-ma-ms-cs.yml \
--ffmpeg 'rtsp://qilei.seu-palm.com:554/live/test' \
--opts MODEL.WEIGHTS ./weights/model_best.pth
```

