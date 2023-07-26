import subprocess as sp
import cv2
import numpy as np
import queue
import threading
from utils.augmentations import letterbox


class RtspVideo(threading.Thread):
    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        super(RtspVideo, self).__init__()
        self.path = path
        self.vid_stride = vid_stride
        self.transforms = transforms
        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
        self.cap = cap
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"fps: {fps}, width: {width}, height: {height}")
        self.queue = queue.Queue(maxsize=400)
        self.exit = False

    def __iter__(self):
        self.start()

    def run(self) -> None:
        while not self.exit:
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            if ret_val:
                if self.transforms:
                    im = self.transforms(im0)  # transforms
                else:
                    im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
                    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    im = np.ascontiguousarray(im)  # contiguous

                self.queue.put((None, im, im0, None, None))
            else:
                self.exit = True
        self.cap.release()

    def __next__(self):
        if self.exit:
            raise StopIteration
        frame = None
        while not frame:
            frame = self.queue.get()
        return frame
