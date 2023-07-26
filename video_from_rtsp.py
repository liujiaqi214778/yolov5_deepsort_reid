import subprocess as sp
import cv2
import sys
import queue
import threading

frame_queue = queue.Queue()

# camera_path='rtsp://172.21.134.74/live/test'
camera_path='rtsp://qilei.seu-palm.com:554/live/test'

# # 获取摄像头参数
# cap = cv2.VideoCapture(camera_path, cv2.CAP_FFMPEG)
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

import numpy as np
import cv2
import threading
from copy import deepcopy

thread_lock = threading.Lock()
thread_exit = False

class myThread(threading.Thread):
    def __init__(self, camera_id, img_height, img_width):
        super(myThread, self).__init__()
        self.camera_id = camera_id
        self.img_height = img_height
        self.img_width = img_width
        self.frame = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    def get_frame(self):
        return deepcopy(self.frame)

    def run(self):
        global thread_exit
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_FFMPEG)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"fps: {fps}, width: {width}, height: {height}, frame count: {frame_count}")
        while not thread_exit:
            ret, frame = cap.read()  # h, w, c
            if ret:
                frame = cv2.resize(frame, (width//2, height//2))
                # thread_lock.acquire()
                # self.frame = frame
                # thread_lock.release()
                frame_queue.put(frame)
            else:
                thread_exit = True
        cap.release()

def main():
    global thread_exit
    camera_id = camera_path
    img_height = 480
    img_width = 640

    # img_height, img_width = height, width
    thread = myThread(camera_id, img_height, img_width)
    thread.start()

    while not thread_exit:
        # thread_lock.acquire()
        # frame = thread.get_frame()
        # thread_lock.release()
        frame = frame_queue.get()

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            thread_exit = True
    thread.join()

if __name__ == "__main__":
    main()
