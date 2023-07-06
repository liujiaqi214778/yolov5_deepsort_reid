import torch
import torch.nn.functional as F
import cv2
import os

# video_fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
video = cv2.VideoWriter("output.mp4", -1, 6, (2560, 1440))
path = os.path.join('.', "pictures_deepsort")
filelist = os.listdir(path)

for file in filelist:
    if file.endswith('jpg'):
        file = os.path.join(path, file)
        img = cv2.imread(file)
        video.write(img)

# 将文件传到windows下再转换成mp4
# linux 下不输出mp4，待解决
video.release()
cv2.destroyAllWindows()
