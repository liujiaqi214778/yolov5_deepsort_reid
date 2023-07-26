import torch
from models.common import DetectMultiBackend
import numpy as np
from utils.general import non_max_suppression, scale_boxes
import cv2

def bbox_r(width, height, *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    margin = min((bbox_left, bbox_top, width - bbox_left - w, height - bbox_top - h))
    return x_c, y_c, w, h, margin


class Person_detect():
    def __init__(self, opt):
        self.device = opt.device if torch.cuda.is_available() else 'cpu'
        self.half = self.device != 'cpu'  # half precision only supported on CUDA
        self.augment = opt.augment
        self.conf_thres = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.classes = opt.classes
        self.agnostic_nms = opt.agnostic_nms
        # Load model
        self.model = DetectMultiBackend(opt.weights, device=self.device, dnn=False, fp16=self.half)  # data=opt.config_yolo
        if self.half:
            self.model.half()
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def detect(self, img, im0s, vid_cap):
        half = self.device != 'cpu'
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=self.augment, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=1000)

        bbox_xywh = []
        confs = []
        clas = []
        xy = []

        for det in pred:
            if det is not None and len(det) != 0:
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    img_h, img_w, _ = im0s.shape
                    x_c, y_c, bbox_w, bbox_h, margin = bbox_r(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    if not conf.item() > self.conf_thres or not int(cls) == 0:
                        continue
                    if obj[2] * obj[3] < 500:  # 像素太低
                        continue
                    # 框在边缘
                    if margin < 8:  # 4
                        continue
                    
                    bbox_xywh.append(obj)
                    confs.append(conf.item())
                    clas.append(cls.item())
                    xy.append(xyxy)
        return np.array(bbox_xywh), confs, clas, xy