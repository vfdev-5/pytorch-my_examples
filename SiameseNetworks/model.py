
import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, Conv2d, MaxPool2d, Sigmoid, ReLU
from torch.autograd import Variable


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Net(Module):
    def __init__(self, input_shape):
        """
        :param input_shape: input image shape, (h, w, c)
        """
        super(Net, self).__init__()

        self.features = Sequential(
            Conv2d(input_shape[-1], 64, kernel_size=10),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2), stride=2),

            Conv2d(64, 128, kernel_size=7),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2), stride=2),

            Conv2d(128, 128, kernel_size=4),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2), stride=2),

            Conv2d(128, 256, kernel_size=4),
            ReLU()
        )

        # Compute number of input features for the last fully-connected layer
        input_shape = (1,) + input_shape[::-1]
        x = Variable(torch.rand(input_shape), requires_grad=False)
        x = self.features(x)
        x = Flatten()(x)
        n = x.size()[1]

        self.classifier = Sequential(
            Flatten(),
            Linear(n, 4096),
            Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SiameseNetworks(Module):
    def __init__(self, input_shape):
        """
        :param input_shape: input image shape, (h, w, c)
        """
        super(SiameseNetworks, self).__init__()
        self.net = Net(input_shape)

        self.classifier = Sequential(
            Linear(4096, 1, bias=False)            
        )
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                m.weight.data.normal_(0, 1e-2)
                m.bias.data.normal_(0.5, 1e-2)
            elif isinstance(m, Linear):
                m.weight.data.normal_(0, 2.0 * 1e-1)
                if m.bias is not None:
                    m.bias.data.normal_(0.5, 1e-2)

    def forward(self, x1, x2):
        x1 = self.net(x1)
        x2 = self.net(x2)
        # L1 component-wise distance between vectors:
        x = torch.pow(torch.abs(x1 - x2), 2.0)
        return self.classifier(x)
