import random
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_class_num(ys):
    from collections import Counter
    num_dict = Counter(ys)
    index = []
    compose = []
    for c in num_dict.keys():
        if num_dict[c] != 0:
            index.append(c)
            compose.append(num_dict[c])
    return index, compose


def classify_label(y, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, label in enumerate(y):
        list1[int(label)].append(idx)
    return list1


class TensorDataset(Dataset):
    def __init__(self, images, labels):  # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]