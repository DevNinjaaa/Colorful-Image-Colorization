import torch.nn as nn
from torchvision.models.resnet import resnet18, ResNet18_Weights

class ColorizationModel(nn.Module):
    def __init__(self, input_size=128):
        super().__init__()
        # Use a pretrained ResNet-18 as the encoder
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Modify the first convolutional layer to accept a single channel (L-channel of LAB)
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the final fully connected layer
        self.encoder.fc = nn.Identity()

        # Decoder to convert from the bottleneck to the AB channels
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 2, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Use sigmoid to scale output to [0, 1] for the AB channels
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 512, 4, 4)
        x = self.decoder(x)
        return x