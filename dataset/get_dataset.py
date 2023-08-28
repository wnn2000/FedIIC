import logging
from collections import Counter
from torchvision.transforms import transforms

from dataset.dataset import SkinDataset, ichDataset
from dataset.randaugment import rand_augment_transform


def get_datasets(args):
    if args.dataset == "isic2019":
        root = "/home/wnn/dataset/isic2019classification/"
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(
            224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
        trans = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform(
                'rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            transforms.ToTensor(),
            normalize,
        ])
        augmentation_sim = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        train_dataset = SkinDataset(root=root, mode="train",
                                    transform=[trans, augmentation_sim])
        val_dataset = SkinDataset(root=root, mode="valid",
                                  transform=val_transform)
        test_dataset = SkinDataset(root=root, mode="test",
                                   transform=val_transform)

    elif args.dataset == "ich":
        root = "/home/wnn/dataset/ICH"
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

        trans1 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        trans2 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

        train_dataset = ichDataset(
            root=root, mode="train", transform=[trans1, trans2])
        val_dataset = ichDataset(
            root=root, mode="valid", transform=val_transform)
        test_dataset = ichDataset(
            root=root, mode="test", transform=val_transform)

    else:
        raise

    logging.info(Counter(train_dataset.targets))
    logging.info(Counter(val_dataset.targets))
    logging.info(Counter(test_dataset.targets))

    return train_dataset, val_dataset, test_dataset

