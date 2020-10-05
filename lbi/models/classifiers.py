import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, width=128, activation=nn.SELU):
        super(ResBlock, self).__init__()
        self.dense = nn.Linear(width, width)
        self.act = activation()
        self.layers = nn.ModuleList([self.dense, self.act])

    def forward(self, x):
        x = x + self.dense(x)
        return self.act(x)


class Unsqueeze(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], 1, -1)


class _Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Classifier(nn.Module):
    def __init__(self, layers=None):
        super().__init__()
        if layers is None:  # do this to avoid default being mutable
            layers = [
                ["Unsqueeze"],
                ["Conv1d", 1, 1, 21],
                ["_Flatten"],
                ["Linear", 320, 100],
                ["ReLU"],
                ["ReLU"],
                ["ResBlock", 100],
                ["Linear", 100, 1],
            ]

        model_layers = []
        for layer_attr in layers:
            layer = getattr(nn, layer_attr[0])(*layer_attr[1:])
            model_layers.append(layer)

        self.layers = model_layers
        self.model = nn.Sequential(*model_layers)

    def forward(self, inputs):
        return self.model(inputs)


# Add custom modules to torch.nn (Probably a really bad idea)
# TODO: Find more elegant and safer alternative
custom_modules = [("ResBlock", ResBlock), ("_Flatten", _Flatten), ("Unsqueeze", Unsqueeze)]
for mod in custom_modules:
    setattr(nn, *mod)


if __name__ == "__main__":
    model = Classifier()
    print(model(torch.rand(10, 340)))
