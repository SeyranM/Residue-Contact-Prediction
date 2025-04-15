import random
from torch.utils.data import Sampler
from typing import List
import numpy as np


class BucketBatchSampler(Sampler):
    def __init__(self, lengths: List[int], batch_size: int, bucket_size: int = 100, shuffle: bool = True):
        self.lengths = np.array(lengths)
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        self.buckets = self._create_buckets()

    def _create_buckets(self):
        sorted_indices = np.argsort(self.lengths)
        buckets = [
            sorted_indices[i:i + self.bucket_size].tolist()
            for i in range(0, len(sorted_indices), self.bucket_size)
        ]
        return buckets

    def __iter__(self):
        if self.shuffle:
            for bucket in self.buckets:
                random.shuffle(bucket)
            random.shuffle(self.buckets)

        for bucket in self.buckets:
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i + self.batch_size]

    def __len__(self):
        return sum(
            len(bucket) // self.batch_size + (1 if len(bucket) % self.batch_size > 0 else 0) for bucket in self.buckets)
