import torch
import torch.nn as nn
import networks.all_models as all_models


class efficientb0(nn.Module):
    def __init__(self, n_classes, args=None):
        super(efficientb0, self).__init__()
        self.n_classes = n_classes
        self.args = args
        self.model = all_models.get_model("Efficient_b0", pretrained=True)
        self.num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(self.num_ftrs, self.n_classes)
        self.projector = nn.Sequential(
            nn.Linear(self.num_ftrs, self.num_ftrs),
            nn.Linear(self.num_ftrs, 1024)
        )

    def forward(self, inputs, project=False):
        if project == False:
            # Convolution layers
            x = self.model.extract_features(inputs)
            # Pooling and final linear layer
            x = self.model._avg_pooling(x)
            x = x.flatten(start_dim=1)
            x = self.model._dropout(x)
            x = self.model._fc(x)
            return x, x
        else:
            # Convolution layers
            x = self.model.extract_features(inputs)
            # Pooling and final linear layer
            x = self.model._avg_pooling(x)
            x = x.flatten(start_dim=1)
            features = self.projector(x)
            y = self.model._dropout(x)
            y = self.model._fc(y)
            return features, y
