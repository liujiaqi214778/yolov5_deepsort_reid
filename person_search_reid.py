import cv2
import torch
import warnings
import argparse
import numpy as np
from utils.dataloaders import LoadImages
# from utils.datasets import LoadImages
from utils.draw import draw_person, draw_boxes
from utils.general import check_img_size
# from person_detect_yolov5 import Person_detect
from person_detect import Person_detect

from utils.parser import get_config
# from torchvision.utils import save_image
# from reid.gallery import Gallery
from deep_sort import DeepReid


class yolo_reid():
    def __init__(self, cfg, args, path):
        self.args = args
        self.video_path = path
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
        # Person_detect行人检测类
        self.person_detect = Person_detect(self.args, self.video_path)
        stride, names, pt = self.person_detect.model.stride, self.person_detect.model.names, self.person_detect.model.pt
        # deepsort 类
        # self.deepsort = build_tracker(cfg, args.sort, use_cuda=use_cuda)
        self.deepsort = DeepReid(cfg, args)

        imgsz = check_img_size(args.img_size, s=stride)  # self.model.stride.max())  # check img_size
        self.dataset = LoadImages(self.video_path, img_size=imgsz, stride=stride, auto=pt, vid_stride=args.vid_stride)
        # self.video_writer = cv2.VideoWriter("output.mp4", -1, 24, (2560, 1440))

    def deep_sort(self):
        idx_frame = 0
        for video_path, img, ori_img, vid_cap, vid_s in self.dataset:
            idx_frame += 1
            if idx_frame % 100 == 0:
                print(f"frame idx: {idx_frame}")

            # if idx_frame < 4300:
            #     continue
            # t1 = time_synchronized()

            # yolo detection
            bbox_xywh, cls_conf, cls_ids, xy = self.person_detect.detect(video_path, img, ori_img, vid_cap)
            if len(cls_conf) == 0:
                if self.args.display:
                    img = cv2.resize(ori_img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
                    cv2.imshow("test", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                # idx = str(idx_frame).rjust(10, '0')
                # cv2.imwrite(f"./pictures/{idx}.jpg", ori_img)
                continue

            # do tracking  # features:reid模型输出2048dim特征
            outputs, features = self.deepsort.update(bbox_xywh, cls_conf, ori_img)

            reid_results = np.arange(len(xy))
            names = np.array([f"yolov5x6" for _ in reid_results])
            ori_img = draw_person(ori_img, xy, reid_results, names)  # draw_person name
            idx = str(idx_frame).rjust(10, '0')
            # cv2.imwrite(f"./pictures/{idx}.jpg", ori_img)

            if len(outputs) > 0:
                # bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_img = draw_boxes(ori_img, bbox_xyxy, identities)
            cv2.imwrite(f"./pictures/{idx}.jpg", ori_img)

                # for bb_xyxy in bbox_xyxy:
                #     bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                # results.append((idx_frame - 1, bbox_tlwh, identities))
            # print("yolo+deepsort:", time_synchronized() - t1)

            if self.args.display:
                img = cv2.resize(ori_img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
                cv2.imshow("test", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default='./ch01_20171125113450.mp4', type=str)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # yolov5
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov5x6.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--classes', default=[0], type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--config-yolo', type=str, default='./yolov5/data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')

    # deep_sort
    parser.add_argument("--sort", default=False, help='True: sort model or False: reid model')
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--display", default=False, help='show resule')
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)

    # reid
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        default='./configs/resnet-bot50-ibn-ma-ms-cs.yml',
        help="path to fastreid config file",
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', './weights/model_best.pth'],
        nargs=argparse.REMAINDER,
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)

    yolo_reid = yolo_reid(cfg, args, path=args.video_path)
    with torch.no_grad():
        yolo_reid.deep_sort()
