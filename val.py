import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


def compute_bacc(model, dataloader, get_confusion_matrix, args):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for (x, label) in dataloader:
            x = x.cuda()
            _, logits = model(x)
            pred = torch.argmax(logits, dim=1)

            all_preds.append(pred.cpu())
            all_labels.append(label)

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    acc = balanced_accuracy_score(all_labels, all_preds)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(all_labels, all_preds)

    if get_confusion_matrix:
        return acc, conf_matrix
    else:
        return acc


def compute_loss(model, dataloader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss = 0.
    with torch.no_grad():
        for (x, label) in dataloader:
            if isinstance(x, list):
                x = x[0]
            x, label = x.cuda(), label.cuda()
            _, logits = model(x)
            loss += criterion(logits, label)
    return loss


def compute_loss_of_classes(model, dataloader, n_classes):
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.eval()

    loss_class = torch.zeros(n_classes).float()
    loss_list = []
    label_list = []

    with torch.no_grad():
        for (x, label) in dataloader:
            if isinstance(x, list):
                x = x[0]
            x, label = x.cuda(), label.cuda()
            _, logits = model(x)
            loss = criterion(logits, label)
            loss_list.append(loss)
            label_list.append(label)

    loss_list = torch.cat(loss_list).cpu()
    label_list = torch.cat(label_list).cpu()

    for i in range(n_classes):
        idx = torch.where(label_list==i)[0]
        loss_class[i] = loss_list[idx].sum()

    return loss_class
