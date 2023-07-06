#!/bin/bash
# Download common models

python -c "
from utils.downloads import attempt_download;
attempt_download('weights/yolov5s.pt');
attempt_download('weights/yolov5m.pt');
attempt_download('weights/yolov5l.pt');
attempt_download('weights/yolov5x.pt');
attempt_download('weights/yolov5x6.pt')
"
