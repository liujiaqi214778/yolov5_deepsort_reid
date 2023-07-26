import os.path
import cv2
import numpy as np
import torch

from person_search_reid import parse_args
from person_detect import Person_detect
from reid.reid_extractor import ReidExtractor
from utils.general import check_img_size
from utils.dataloaders import LoadImages


def xywh_to_xyxy(bbox_xywh, W, H):
    x, y, w, h = bbox_xywh
    x1 = max(int(x - w / 2), 0)
    x2 = min(int(x + w / 2), W - 1)
    y1 = max(int(y - h / 2), 0)
    y2 = min(int(y + h / 2), H - 1)
    return x1, y1, x2, y2


def make_gallery(args, path, output_path):
    assert path != output_path
    detector = Person_detect(args)
    extractor = ReidExtractor(args)
    stride, names, pt = detector.model.stride, detector.model.names, detector.model.pt
    imgsz = check_img_size(args.img_size, s=stride)
    dataset = LoadImages(path, img_size=imgsz, stride=stride, auto=pt, vid_stride=args.vid_stride)

    names, im_crops = [], []
    pid = -1
    for file_path, img, ori_img, vid_cap, vid_s in dataset:
        bbox_xywh, cls_conf, cls_ids, xy = detector.detect(img, ori_img, vid_cap)
        if len(cls_conf) == 0:
            continue
        pid += 1
        area = bbox_xywh[:, 2] * bbox_xywh[:, 3]  # w * h
        person_bbox = bbox_xywh[area.argmax()]

        file_name = os.path.basename(file_path)
        H, W = ori_img.shape[:2]
        x1, y1, x2, y2 = xywh_to_xyxy(person_bbox, W, H)
        im = ori_img[y1:y2, x1:x2]
        spid = str(pid).rjust(6, '0')
        cv2.imwrite(os.path.join(output_path, spid + '_' + file_name), im)

        person_name = file_name.split('.')[0]
        names.append(person_name)
        im = im[:, :, ::-1]  # reid 前处理
        im = cv2.resize(im, (128, 256), interpolation=cv2.INTER_CUBIC)
        im_crops.append(torch.as_tensor(im.astype("float32").transpose(2, 0, 1))[None])

    if im_crops:
        features = extractor(im_crops)
        np.save(os.path.join(output_path, 'names.npy'), np.array(names))
        np.save(os.path.join(output_path, 'gallery.npy'), features.numpy())


if __name__ == '__main__':
    args = parse_args()
    # deepsort_cfg = get_config()
    # deepsort_cfg.merge_from_file(args.config_deepsort)
    output_path = os.path.join('.', 'reid', 'gallery')
    path = os.path.join('.', 'pictures')
    make_gallery(args, path, output_path)
