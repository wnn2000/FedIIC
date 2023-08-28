import os
from tqdm import tqdm
import argparse
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from networks.networks import efficientb0
from dataset.get_dataset import get_datasets
from utils.utils import set_seed


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='isic2019', help='dataset name')
    parser.add_argument('--exp', type=str,
                        default='', help='experiment name')
    parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
    parser.add_argument('--mode', type=str,
                        default='test', help='test or valid')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = args_parser()
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ------------------------------ deterministic or not ------------------------------
    cudnn.benchmark = False
    cudnn.deterministic = True
    set_seed(args)

    # ------------------------------ dataset and dataloader ------------------------------
    _, _, test_dataset = get_datasets(args)
    print(Counter(test_dataset.targets))
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )
    
    # ------------------------------ load model ------------------------------
    model = efficientb0(n_classes=test_dataset.n_classes).cuda()
    model_dir = os.path.join('outputs', args.exp,
                             'models', 'best_model.pth')
    model.load_state_dict(torch.load(model_dir))
    print("have loaded the best model from {}".format(model_dir))


    # ------------------------------ test or valid ------------------------------
    np.set_printoptions(suppress=True)

    all_preds = []
    all_labels = []
    all_prob = []
    model.eval()
    with torch.no_grad():
        for (x, label) in tqdm(test_loader):
            x = x.cuda()
            _, logits = model(x)
            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)

            all_prob.append(prob.cpu())
            all_preds.append(pred.cpu())
            all_labels.append(label)

    all_prob = torch.cat(all_prob).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(conf_matrix)

    result = classification_report(all_labels, all_preds, digits=7)
    print(result)

