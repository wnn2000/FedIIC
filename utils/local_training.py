import logging
import numpy as np
import copy

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils.losses import IntraSCL, InterSCL, DALA


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

    def get_num_class_list(self):
        self.n_classes = self.dataset.n_classes
        class_num = np.array([0] * self.n_classes)
        for idx in self.idxs:
            label = self.dataset.targets[idx]
            class_num[label] += 1
        return class_num.tolist()


class LocalUpdate(object):
    def __init__(self, args, id, dataset, idxs):
        self.args = args
        self.id = id
        self.local_dataset = DatasetSplit(dataset, idxs)
        self.class_num_list = self.local_dataset.get_num_class_list()
        logging.info(
            f"Client{id} ===> Each class num: {self.class_num_list}, Total: {len(self.local_dataset)}")
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        self.epoch = 0
        self.iter_num = 0
        self.lr = self.args.base_lr


    def train(self, net, writer):
        net.train()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"Id: {self.id}, Num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        ce_criterion = nn.CrossEntropyLoss()
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (images, labels) in self.ldr_train:
                images, labels = images.cuda(), labels.cuda()

                _, logits = net(images)
                loss = ce_criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return net.state_dict(), np.array(epoch_loss).mean()


    def train_FedIIC(self, net, prototypes, writer):
        net.train()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"Id: {self.id}, Num: {len(self.local_dataset)}")

        prototypes = F.normalize(prototypes, dim=1).detach().clone().cuda()

        intra_cl_criterion = IntraSCL(cls_num_list=self.class_num_list)
        inter_cl_criterion = InterSCL(cls_num_list=self.class_num_list)
        ce_criterion = DALA(
            cls_num_list=self.class_num_list, cls_loss=self.loss_class, epoch=self.epoch, args=self.args)

        # train and update
        epoch_loss = []
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (images, labels) in self.ldr_train:
                assert isinstance(images, list)
                images = torch.cat([images[0], images[1]], dim=0)
                images, labels = images.cuda(), labels.cuda()

                features, logits = net(images, project=True)

                features = F.normalize(features, dim=1)

                split_size = features.size()[0] // 2

                f1, f2 = torch.split(features, [split_size, split_size], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                logits, _ = torch.split(
                    logits, [split_size, split_size], dim=0)

                loss_ce = ce_criterion(logits, labels)
                loss_cl_inter = inter_cl_criterion(prototypes, features, labels)
                loss_cl_intra = intra_cl_criterion(features, labels)

                loss = loss_ce + self.args.k1*loss_cl_intra + self.args.k2*loss_cl_inter

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss', loss.item(), self.iter_num)
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss_ce, self.iter_num)
                writer.add_scalar(
                    f'client{self.id}/loss_cl_inter', loss_cl_inter, self.iter_num)
                writer.add_scalar(
                    f'client{self.id}/loss_cl_intra', loss_cl_intra, self.iter_num)

                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return net.state_dict(), np.array(epoch_loss).mean()
