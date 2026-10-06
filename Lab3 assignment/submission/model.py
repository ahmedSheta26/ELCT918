# model.py -- LeNet-5 for MNIST (1-channel, 32x32 input, 10 classes)

import torch.nn as nn  # nn contains the building blocks of neural networks (layers, etc.)


class LeNet5(nn.Module):
    # nn.Module is PyTorch's base class for every model. Inheriting from it
    # lets PyTorch find all our layers' weights automatically (needed for training and saving).

    def __init__(self, num_classes=10):
        # __init__ runs once when we create the model. It only DEFINES the layers.
        super().__init__()  # required: initialises the nn.Module machinery

        # "features" = the part that looks at the image and extracts patterns
        self.features = nn.Sequential(  # Sequential = run layers one after another
            # C1: 1 input channel (grayscale) -> 6 feature maps, using 5x5 filters.
            # A "filter" is a small grid of learnable numbers slid over the image
            # to detect simple patterns (edges, curves). Size: 32x32 -> 28x28.
            nn.Conv2d(1, 6, kernel_size=5),   # <-- the ONLY layer changed (3 -> 1)
            nn.Tanh(),  # activation: squashes values into (-1, 1); lets the network learn curved, non-linear rules
            # S2: average each 2x2 block -> halves the size: 28x28 -> 14x14.
            nn.AvgPool2d(kernel_size=2, stride=2),

            # C3: 6 maps -> 16 maps with 5x5 filters. Size: 14x14 -> 10x10.
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            # S4: 10x10 -> 5x5.
            nn.AvgPool2d(kernel_size=2, stride=2),

            # C5: 16 maps -> 120 maps with 5x5 filters. The input is 5x5, so the output
            # is 1x1: each of the 120 outputs "sees" the whole image.
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Tanh(),
        )

        # "classifier" = the part that decides which digit it is
        self.classifier = nn.Sequential(
            nn.Flatten(),             # turn (120,1,1) into a flat list of 120 numbers
            nn.Linear(120, 84),       # fully connected layer: every input connects to every output
            nn.Tanh(),
            nn.Linear(84, num_classes),  # 10 outputs, one score per digit ("logits")
            # No softmax here: the loss function we use applies it internally.
        )

    def forward(self, x):
        # forward = how data flows through the model: image -> features -> classifier
        return self.classifier(self.features(x))


def count_params(model):
    # p.numel() = number of values in a weight tensor. We add them all up,
    # counting only the ones that are learned (requires_grad = True).
    return sum(p.numel() for p in model.parameters() if p.requires_grad)