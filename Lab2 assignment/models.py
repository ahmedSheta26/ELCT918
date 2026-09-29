"""
models.py  --  Lab 2, Task 1
LeNet-5, AlexNet and VGG16 implemented in PyTorch for 32x32 CIFAR inputs.

All three models:
  * take a tensor of shape (batch, 3, 32, 32)
  * return raw scores ("logits") of shape (batch, num_classes)
    -> do NOT add softmax; nn.CrossEntropyLoss does it for you
  * are built from scratch (no pretrained weights)

Usage:
    from models import get_model
    model = get_model("vgg16", num_classes=10)
"""

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# 1) LeNet-5   (LeCun et al., 1998)
# --------------------------------------------------------------------------
class LeNet5(nn.Module):
    """
    Input 3x32x32
    C1: conv 6@5x5            -> 6x28x28
    S2: avg-pool 2x2, stride2 -> 6x14x14
    C3: conv 16@5x5           -> 16x10x10
    S4: avg-pool 2x2, stride2 -> 16x5x5
    C5: conv 120@5x5          -> 120x1x1   (acts like a fully-connected layer)
    F6: fc 84
    Out: fc num_classes

    Adaptations (state these in your report):
      * input has 3 channels (RGB) instead of 1 (grayscale digits)
      * C3 is fully connected to all 6 maps (the paper used a sparse
        connection table to save computation)
      * output layer is a plain linear layer instead of RBF units
      * tanh + average pooling kept, as in the paper
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),      # C1
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),   # S2
            nn.Conv2d(6, 16, kernel_size=5),     # C3
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),   # S4
            nn.Conv2d(16, 120, kernel_size=5),   # C5
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),                  # F6
            nn.Tanh(),
            nn.Linear(84, num_classes),          # output
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------
# 2) AlexNet   (Krizhevsky et al., 2012)
# --------------------------------------------------------------------------
class AlexNet(nn.Module):
    """
    Input 3x32x32
    conv1 96@5x5, s1, p2 + ReLU + LRN + maxpool 3x3/s2 -> 96x15x15
    conv2 256@5x5, p2    + ReLU + LRN + maxpool 3x3/s2 -> 256x7x7
    conv3 384@3x3, p1    + ReLU                        -> 384x7x7
    conv4 384@3x3, p1    + ReLU                        -> 384x7x7
    conv5 256@3x3, p1    + ReLU + maxpool 3x3/s2       -> 256x3x3
    fc6 4096 + ReLU + Dropout(0.5)
    fc7 4096 + ReLU + Dropout(0.5)
    fc8 num_classes

    Adaptations (state these in your report):
      * conv1 is 5x5 stride 1 instead of 11x11 stride 4: an 11x11/stride-4
        filter would shrink a 32x32 image to ~7x7 immediately and throw away
        most of the information
      * fc6 input is 256*3*3 = 2304 (the paper has 256*6*6 = 9216)
      * fc8 has num_classes outputs instead of 1000
      * the paper's 2-GPU split (filters only see half the previous maps)
        is not used; each layer is one ordinary convolution
    Kept from the paper: channel widths, ReLU, overlapping max-pooling,
    local response normalisation, dropout 0.5, 4096-4096 FC layers.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------
# 3) VGG16   (Simonyan & Zisserman, 2014)  -- configuration "D"
# --------------------------------------------------------------------------
# numbers = conv output channels (all 3x3, stride 1, padding 1), "M" = maxpool 2x2/s2
VGG16_CFG = [64, 64, "M",
             128, 128, "M",
             256, 256, 256, "M",
             512, 512, 512, "M",
             512, 512, 512, "M"]


class VGG16(nn.Module):
    """
    13 conv layers + 3 FC layers = 16 weight layers.
    On a 32x32 input the 5 max-pools reduce the map 32->16->8->4->2->1,
    so the flattened feature vector has 512*1*1 = 512 values
    (the paper's 224x224 input gives 512*7*7 = 25088).

    Adaptations (state these in your report):
      * fc6 input is 512 instead of 25088
      * fc8 has num_classes outputs instead of 1000
    Kept from the paper: every conv layer, 4096-4096 FC layers, dropout 0.5.

    use_bn=False is the faithful version. If your loss stays stuck at
    ~2.303 (CIFAR-10) or ~4.605 (CIFAR-100) and accuracy never leaves
    chance level, set use_bn=True and say so in the report.
    """

    def __init__(self, num_classes=10, use_bn=False):
        super().__init__()
        layers, in_ch = [], 3
        for v in VGG16_CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_ch, v, kernel_size=3, padding=1))
                if use_bn:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def init_weights(model):
    """Kaiming init for ReLU nets (helps deep nets like VGG16 start learning)."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)


def get_model(name, num_classes=10, **kwargs):
    name = name.lower()
    models = {"lenet": LeNet5, "alexnet": AlexNet, "vgg16": VGG16}
    model = models[name](num_classes=num_classes, **kwargs)
    if name != "lenet":          # LeNet uses tanh; PyTorch's default init suits it
        init_weights(model)
    return model


def count_params(model):
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------
# Sanity check:  python models.py
# Prints every layer's output shape and each network's parameter count.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)            # a fake batch of 2 images
    for name in ["lenet", "alexnet", "vgg16"]:
        for n_cls in [10, 100]:
            model = get_model(name, n_cls)
            model.eval()
            out = model(x)
            assert out.shape == (2, n_cls), out.shape
            print(f"{name:8s} classes={n_cls:3d}  output={tuple(out.shape)}  "
                  f"params={count_params(model):,}")

        # layer-by-layer shapes (useful for your Task 2 table)
        print(f"\n--- {name} layer-by-layer (10 classes) ---")
        model = get_model(name, 10).eval()
        h = x
        for layer in list(model.features) + list(model.classifier):
            h = layer(h)
            if not isinstance(layer, (nn.ReLU, nn.Tanh, nn.Dropout)):
                print(f"{layer.__class__.__name__:22s} -> {tuple(h.shape[1:])}")
        print()
