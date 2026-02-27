import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_cnn=False):
        """Init encoder CNN."""
        super(EncoderCNN, self).__init__()

        # load ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # remove fc layer
        modules = list(self.resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # freeze CNN
        if not train_cnn:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # 2048 to embed
        self.linear = nn.Linear(2048, embed_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        """Forward pass."""

        # spatial features
        features = self.resnet(images)

        batch_size = features.size(0)

        # reshape features
        features = features.view(batch_size, 2048, -1)
        features = features.permute(0, 2, 1)

        # project features
        features = self.linear(features)
        features = self.relu(features)
        features = self.dropout(features)

        return features
