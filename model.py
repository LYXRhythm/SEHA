import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, tanh: bool = True):
        super(FeatureExtractor, self).__init__()
        self.tanh = tanh
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim, bias=False)
        )
        self._init_weights(input_dim, output_dim)

    def _init_weights(self, input_dim: int, output_dim: int):
        nn.init.uniform_(
            self.layers[-1].weight, 
            -1. / np.sqrt(np.float32(input_dim)), 
            1. / np.sqrt(np.float32(input_dim))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.layers(x)
        if self.tanh:
            features = torch.tanh(features)
        
        norm = torch.norm(features, p=2, dim=1, keepdim=True)
        return features / norm

class Embedding(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int):
        super(Embedding, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(num_classes, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, embedding_dim)
        )
        
        self._init_weights(embedding_dim, num_classes)

    def _init_weights(self, embedding_dim: int, num_classes: int):
        nn.init.uniform_(
            self.layers[-1].weight, 
            -1. / np.sqrt(np.float32(num_classes)), 
            1. / np.sqrt(np.float32(num_classes))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.layers(x)
        embeddings = torch.tanh(embeddings)
        
        norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        return embeddings / norm

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()

        vgg = models.vgg19_bn(pretrained=True)
        self.features = vgg.features
        self.classifier_features = nn.Sequential(*list(vgg.classifier.children())[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_features = self.features(x)
        flattened = conv_features.view(x.size(0), -1)
        final_features = self.classifier_features(flattened)
        return final_features

class CMNN(nn.Module):
    def __init__(self, img_input_dim: int = 4096, text_input_dim: int = 1024, 
                 output_dim: int = 1024, num_class: int = 10, tanh: bool = True):
        super(CMNN, self).__init__()
        
        self.img_net = FeatureExtractor(img_input_dim, output_dim, tanh)
        self.text_net = FeatureExtractor(text_input_dim, output_dim, tanh)

    def forward(self, img: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img_feature = self.img_net(img)
        text_feature = self.text_net(text)
        return img_feature, text_feature