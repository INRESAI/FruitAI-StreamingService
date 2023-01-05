import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as func


class Net(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.fc_class = nn.Sequential(
            nn.Linear(in_features=1408, out_features=3, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.model.global_pool(x)
        if self.model.drop_rate > 0.:
            x = func.dropout(x, p=self.model.drop_rate, training=self.model.training)
        y_class = self.fc_class(x)
        return y_class


def load_model(weight):
    model_efficent = timm.create_model('tf_efficientnetv2_b2', pretrained=True)
    model = Net(model_efficent)
    check_point_cls = torch.load(weight)
    model.load_state_dict(check_point_cls)
    model.to('cuda')
    model.eval()
    return model


def classify(image, model):
    img = cv2.resize(image, (256, 256))
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.tensor(img, dtype=torch.float).to('cuda') 

    output = model(img)
    index_label = output.detach().cpu().numpy()[0]
    index = np.argmax(index_label)
    return index
