import os
import sys
import copy
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.FedAvg import FedAvg
from dataset.get_dataset import get_datasets
from val import compute_bacc, compute_loss_of_classes
from networks.networks import efficientb0
from utils.local_training import LocalUpdate
from utils.utils import set_seed, TensorDataset, classify_label
from utils.sample_dirichlet import clients_indices


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='isic2019', help='dataset name')
    parser.add_argument('--exp', type=str,
                        default='FedIIC', help='experiment name')
    parser.add_argument('--batch_size', type=int,
                        default=2, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float,  default=3e-4,
                        help='base learning rate')
    parser.add_argument('--alpha', type=float,
                        default=1.0, help='parameter for non-iid')
    parser.add_argument('--k1', type=float,  default=2.0,
                        help='weight for Intra-client contrastive learning')
    parser.add_argument('--k2', type=float,  default=2.0,
                        help='weight for Inter-client contrastive learning')
    parser.add_argument('--d', type=float,  default=0.25,
                        help='difficulty')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
    parser.add_argument('--local_ep', type=int,
                        default=1, help='local epoch')
    parser.add_argument('--rounds', type=int,  default=200, help='rounds')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # ------------------------------ deterministic or not ------------------------------
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args)

    # ------------------------------ output files ------------------------------
    outputs_dir = 'outputs'
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)
    exp_dir = os.path.join(outputs_dir, args.exp + '_' + '_' + str(args.local_ep) + '_' + str(args.k1) + '_' + str(args.k2))
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    models_dir = os.path.join(exp_dir, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    logs_dir = os.path.join(exp_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    tensorboard_dir = os.path.join(exp_dir, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)

    logging.basicConfig(filename=logs_dir+'/logs.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    writer = SummaryWriter(tensorboard_dir)

    # ------------------------------ dataset and dataloader ------------------------------
    train_dataset, val_dataset, test_dataset = get_datasets(args)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    if args.dataset == "isic2019":
        args.n_clients = 10
    elif args.datasets == "ich":
        args.n_clients = 20
    else:
        raise

    # ------------------------------ global and local settings ------------------------------
    n_classes = train_dataset.n_classes
    net_glob = efficientb0(n_classes=n_classes, args=args).cuda()
    net_glob.train()
    w_glob = net_glob.state_dict()
    w_locals = []
    trainer_locals = []
    net_locals = []
    user_id = list(range(args.n_clients))

    # Here, we follow CreFF (https://arxiv.org/abs/2204.13399).
    list_label2indices = classify_label(train_dataset.targets, n_classes)
    dict_users = clients_indices(list_label2indices, n_classes, args.n_clients, args.alpha, args.seed)
    dict_len = [len(dict_users[id]) for id in user_id]

    for id in user_id:
        trainer_locals.append(LocalUpdate(
            args, id, copy.deepcopy(train_dataset), dict_users[id]))
        w_locals.append(copy.deepcopy(w_glob))
        net_locals.append(copy.deepcopy(net_glob).cuda())

    images_all = {}
    labels_all = {}
    for id in user_id: # to compute loss quickly
        local_set = copy.deepcopy(trainer_locals[id].local_dataset)
        images_all[id] = torch.cat([torch.unsqueeze(local_set[i][0][0], dim=0)
                                    for i in range(len(local_set))])
        labels_all[id] = torch.tensor([int(local_set[i][1])
                                       for i in range(len(local_set))]).long()
        print(id, ':', len(images_all[id]), labels_all[id])

    # ------------------------------ begin training ------------------------------
    set_seed(args)
    best_performance = 0.
    lr = args.base_lr
    acc = []
    for com_round in range(args.rounds):
        logging.info(f'\n======================> round: {com_round} <======================')
        loss_locals = []
        writer.add_scalar('train/lr', lr, com_round)

        with torch.no_grad():
            class_embedding = w_glob["model._fc.weight"].detach().clone().cuda()
            feature_avg = net_glob.projector(class_embedding).detach().clone()
        print("similarity before")
        print(torch.matmul(F.normalize(feature_avg, dim=1),
              F.normalize(feature_avg, dim=1).T))
        feature_avg.requires_grad = True
        optimizer_f = torch.optim.SGD([feature_avg], lr=0.1)
        mask = torch.ones((n_classes, n_classes)) - torch.eye((n_classes))
        mask = mask.cuda()
        for i in range(1000):
            feature_avg_n = F.normalize(feature_avg, dim=1)
            cos_sim = torch.matmul(feature_avg_n, feature_avg_n.T)
            cos_sim = ((cos_sim * mask).max(1)[0]).sum()
            optimizer_f.zero_grad()
            cos_sim.backward()
            optimizer_f.step()
        print("similarity after")
        print(torch.matmul(F.normalize(feature_avg, dim=1),
              F.normalize(feature_avg, dim=1).T))

        loss_matrix = torch.zeros(args.n_clients, n_classes)
        class_num = torch.zeros(args.n_clients, n_classes)
        net_glob = net_glob.cuda()
        for id in user_id:
            class_num[id] = torch.tensor(
                trainer_locals[id].local_dataset.get_num_class_list())
            dataset_client = TensorDataset(images_all[id], labels_all[id])
            dataLoader_client = DataLoader(
                dataset_client, batch_size=32, shuffle=False)
            loss_matrix[id] = compute_loss_of_classes(
                net_glob, dataLoader_client, n_classes)
        num = torch.sum(class_num, dim=0, keepdim=True)
        logging.info("class-num of this round")
        logging.info(num)
        loss_matrix = loss_matrix / (1e-5 + num)
        loss_class = torch.sum(loss_matrix, dim=0)
        logging.info("loss of this round")
        logging.info(loss_class)

        # local training
        for id in user_id:
            trainer_locals[id].lr = lr
            local = trainer_locals[id]
            local.loss_class = loss_class
            net_local = net_locals[id]
            w, loss = local.train_FedIIC(copy.deepcopy(
                net_local), copy.deepcopy(feature_avg), writer)
            w_locals[id] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))

        # upload and download
        with torch.no_grad():
            w_glob = FedAvg(w_locals, dict_len)
        net_glob.load_state_dict(w_glob)
        for id in user_id:
            net_locals[id].load_state_dict(w_glob)

        # global validation
        net_glob = net_glob.cuda()
        bacc_g, conf_matrix = compute_bacc(
            net_glob, val_loader, get_confusion_matrix=True, args=args)
        writer.add_scalar(
            f'glob/bacc_val', bacc_g, com_round)
        logging.info('global conf_matrix')
        logging.info(conf_matrix)

        # save model
        if bacc_g > best_performance:
            best_performance = bacc_g
            torch.save(net_glob.state_dict(),  models_dir +
                       f'/best_model_{com_round}_{best_performance}.pth')
            torch.save(net_glob.state_dict(),  models_dir+'/best_model.pth')
        logging.info(f'best bacc: {best_performance}, now bacc: {bacc_g}')
        acc.append(bacc_g)

    writer.close()
    logging.info(acc)
