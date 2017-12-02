
import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, Conv2d, MaxPool2d, Sigmoid, ReLU, Dropout
from torch.autograd import Variable


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Net(Module):
    def __init__(self, input_shape, with_dropout=False):
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

        classifier = [
            Flatten(),
            Linear(n, 4096),
            Sigmoid()
        ]

        if with_dropout:
            classifier.insert(1, Dropout(p=0.5))

        self.classifier = Sequential(*classifier)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SiameseNetworks(Module):
    def __init__(self, input_shape, with_dropout=False, distance_type='L1'):
        """
        :param input_shape: input image shape, (h, w, c)
        :param distance_type: component-wise distance between feature vectors produced by a single CNN
            Default, is L1 distance (used in the paper)
            Possible values: 'L1', 'Cosine'
        """
        assert distance_type in ['L1', 'Cosine']
        super(SiameseNetworks, self).__init__()
        self.net = Net(input_shape, with_dropout)

        classifier = [Linear(4096, 1, bias=False)]
        if with_dropout:
            classifier.insert(0, Dropout(p=0.5))

        self.classifier = Sequential(*classifier)
        self._weight_init()

        if distance_type == 'L1':
            self.dist_fn = torch.abs
        elif distance_type == 'Cosine':
            self.dist_fn = torch.cos

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
        # Component-wise distance between vectors:
        x = self.dist_fn(x1 - x2)
        return self.classifier(x)
