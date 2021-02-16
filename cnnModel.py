from torch import nn
from torch.autograd import Variable
import torch

class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=9, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 100, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(160 * 100, 80 * 100),
            nn.BatchNorm1d(80 * 100),
            nn.ReLU(),
            nn.Dropout()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(80 *100, 40 * 100),
            nn.BatchNorm1d(40 * 100),
            nn.ReLU(),
            nn.Dropout()
        )
        self.layer6 = nn.Sequential(
            nn.Linear(40 * 100, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU(),
            nn.Dropout()
        )
        self.layer7 = nn.Sequential(
            nn.Linear(2000,  1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout()
        )
        self.layer8 = nn.Sequential(
            nn.Linear(1000, 30)
        )

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = nn.functional.log_softmax(out, 1)
        return out
