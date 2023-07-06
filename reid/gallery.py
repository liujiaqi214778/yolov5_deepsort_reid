import torch
import os
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from fastreid.utils.compute_dist import build_dist


@torch.no_grad()
def cosine_distance(features, others):
    """Computes cosine distance.
    Args:
        features (torch.Tensor): 2-D feature matrix.
        others (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    features = F.normalize(features, p=2, dim=1)
    others = F.normalize(others, p=2, dim=1)
    dist_m = 1 - torch.mm(features, others.t())
    return dist_m


class Gallery:
    def __init__(self, cfg):
        """
        每个pid在gallery只有一个feature
        """
        path = os.path.join("reid", "gallery")
        self.g_path = os.path.join(path, "gallery.npy")
        self.names_path = os.path.join(path, "names.npy")
        self.gallery = torch.from_numpy(np.load(self.g_path)).cuda()
        self.names = np.load(self.names_path)  # gallery 实例对应的id或name

        self.thres = 0.6  # 特征检索阈值  (0, 2)，最大余弦距离
        self.maxn = 100  # gallery中最大人物数
        self.idn = 1  # 每个人物特征数
        self.pid_new = len(self.gallery)

    def search(self, query):
        if len(query.size()) == 1:
            query = query.unsqueeze(0)
        num_q, num_g = len(query), len(self.gallery)
        assert num_q > 0
        query = query.cuda()
        if len(self.gallery) > 0:
            dist = cosine_distance(query, self.gallery)  # np.ndarray
            # indices = np.argsort(dist, axis=1)
            # indices = torch.argmin(dist, dim=1)
            minimum, indices = dist.min(dim=1)
            q = query[minimum > self.thres]
            # 一张图像中检索出两个相同pid？
            indices[minimum > self.thres] = torch.arange(num_g, num_g + len(q)).cuda() if self.update(q) else -1
        else:
            q = query[:self.maxn]
            indices = torch.arange(min(len(q), self.maxn)).cuda()
            self.update(q)

        return indices, self.names

    def update(self, features: torch.Tensor, pids=None) -> bool:
        if len(features.size()) == 1:
            features = features.unsqueeze(0)
        n = len(features)
        if pids is None:
            if n > 0 and len(self.gallery) + n <= self.maxn:
                pids = np.array([self.generate_pid() for _ in range(n)])
                self.gallery = torch.cat((self.gallery, features))
                self.names = np.concatenate((self.names, pids))
            else:
                return False
        else:
            assert len(pids) == n
            self.gallery[pids] = features
        return True

    def generate_pid(self):
        pid = f"person_{self.pid_new}"
        self.pid_new += 1
        return pid

    def __len__(self):
        return len(self.gallery)

    def save(self):
        np.save(self.g_path, self.gallery.cpu().numpy())
        np.save(self.names_path, self.names)
