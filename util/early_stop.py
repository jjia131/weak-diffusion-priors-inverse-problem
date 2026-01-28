import numpy as np
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except ImportError:  # torch is optional for numpy-only pipelines
    torch = None

class EarlyStop():
    def __init__(self, size, patience):
        self.patience = patience
        self.wait_count = 0
        self.best_score = float('inf')
        self.best_epoch = 0
        self.img_collection = []
        self.stop = False
        self.size = size

    def check_stop(self, current, cur_epoch):
      #stop when variance doesn't decrease for consecutive P(patience) times
        if current < self.best_score:
            self.best_score = current
            self.best_epoch = cur_epoch
            self.wait_count = 0
            should_stop = False
        else:
            self.wait_count += 1
            should_stop = self.wait_count >= self.patience
        return should_stop

    def update_img_collection(self, cur_img):
        self.img_collection.append(cur_img)
        if len(self.img_collection) > self.size:
            self.img_collection.pop(0)

    def get_img_collection(self):
        return self.img_collection

def myMetric(x1, x2):
    return ((x1 - x2) ** 2).sum() / x1.size
    # return (np.abs(x1 - x2)).sum() / x1.size


def check_early_stop(earlystop, new_sample, cur_epoch):
    #variance hisotry for early stop
    earlystop.update_img_collection(new_sample)
    img_collection = earlystop.get_img_collection()
    if len(img_collection) == earlystop.size:
        ave_img = np.mean(img_collection, axis = 0)
        variance = []
        for tmp in img_collection:
            variance.append(myMetric(ave_img, tmp))
        cur_var = np.mean(variance)
        if earlystop.stop == False:
            earlystop.stop = earlystop.check_stop(cur_var, cur_epoch)

class SimpleHoldOutStop:
    """
    Minimal hold-out stopper that tracks the smallest `best_k` validation
    losses and stops once no improvement happens for `patience` iterations.
    The best epoch is defined as the most recent epoch among the retained
    top-k candidates, mirroring the notebook heuristic.
    """

    def __init__(self, patience: int = 1000, best_k: int = 1):
        if patience <= 0:
            raise ValueError("patience must be positive")
        if best_k <= 0:
            raise ValueError("best_k must be positive")
        self.patience = patience
        self.best_k = best_k
        self._wait = 0
        self._topk: List[Tuple[float, int]] = []
        self.best_epoch: Optional[int] = None

    def update(self, val_loss: float, epoch: int) -> Tuple[bool, Optional[int]]:
        """
        Record a new validation loss.

        Returns:
            should_stop: True if the patience budget is exhausted.
            best_epoch: Latest epoch among the current best_k candidates.
        """
        improved = self._try_insert(val_loss, epoch)
        if improved:
            self._wait = 0
        else:
            self._wait += 1
        should_stop = self._wait >= self.patience
        self.best_epoch = self._latest_epoch()
        return should_stop, self.best_epoch

    def _try_insert(self, val_loss: float, epoch: int) -> bool:
        if len(self._topk) < self.best_k:
            self._topk.append((val_loss, epoch))
            self._topk.sort(key=lambda item: (item[0], item[1]))
            return True
        worst_idx = max(range(len(self._topk)), key=lambda i: self._topk[i][0])
        worst_loss, _ = self._topk[worst_idx]
        if val_loss < worst_loss:
            self._topk[worst_idx] = (val_loss, epoch)
            self._topk.sort(key=lambda item: (item[0], item[1]))
            return True
        return False

    def _latest_epoch(self) -> Optional[int]:
        if not self._topk:
            return None
        return max(epoch for _, epoch in self._topk)
