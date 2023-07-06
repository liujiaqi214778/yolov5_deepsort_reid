import torch

from .track import Track, TrackState
from .tracker import Tracker
from reid.gallery import Gallery
import numpy as np


class ReIDTrack(Track):
    def update_with_gallery(self, kf, detection, gallery: Gallery, confirmed_ids: set):
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
            feature = torch.from_numpy(detection.feature).unsqueeze(0).cuda()
            idx, names = gallery.search(feature)
            idx = int(idx)  # idx 可能为 -1, pid不存在
            if idx != -1 and idx in confirmed_ids:  # search到了但是当前帧已有这个id
                if gallery.update(feature):
                    idx = len(gallery) - 1
                else:
                    idx = -1
            confirmed_ids.add(idx)
            self.track_id = idx
        else:
            pass  # 是否更新gallery里已存在的feature


class ReIDTracker(Tracker):
    def __init__(self, gallery: Gallery, metric, **kwargs):
        super().__init__(metric, **kwargs)
        self.gallery = gallery
        self.confirmed_ids = set()

    def update(self, detections):
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        # Update track set.
        self.confirmed_ids = set([t.track_id for t in self.tracks if t.is_confirmed()])
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update_with_gallery(
                self.kf, detections[detection_idx], self.gallery, self.confirmed_ids)

        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        # idx, names = self.gallery.search(detection.feature)
        # idx = int(idx)  # idx 可能为 -1, pid不存在
        # # if idx == -1:
        # #     return
        # if idx in self.confirmed_ids:  # search到了但是当前帧已有这个id
        #     self.gallery.update(detection.feature)
        #     idx = len(self.gallery) - 1

        self.tracks.append(ReIDTrack(
            mean, covariance, -1, self.n_init, self.max_age,
            detection.feature))