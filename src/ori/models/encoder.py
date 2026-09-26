import torch.nn as nn
import torch.nn.functional as F
import torchvision



class encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(encoder, self).__init__()

        resnet = torchvision.models.resnet50(weights='DEFAULT' if pretrained else None)

        self.base_model = nn.Sequential(*list(resnet.children())[:-1])

        self.embedding = nn.Linear(2048, 1024)

        self.projection = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128)
        )


    def forward(self, x):
        z = self.base_model(x)
        z = z.squeeze()
        z = self.embedding(z)

        z = self.projection(z)
        z = F.normalize(z, dim=1)
        return z