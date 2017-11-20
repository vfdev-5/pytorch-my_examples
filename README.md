# PyTorch framework tutorials

*My examples to start training with PyTorch*

## Content

- *Basics*: basic examples of tensor manipulations with PyTorch

- *Cifar10* Classification 
    - Setup dataflow
    - Setup model
    - Training
   
- *Siamese Networks* one-shot learning
    - Omniglot dataflow
    - 

## Dependencies

Notebooks use Python3 kernel

### PyTorch
To run tutorial notebooks, install 
```
pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl
pip3 install torchvision
```
for other platforms/options see [official site](pytorch.org)


### CIFAR10

We use CIFAR10 dataset in the tutorial (python batches) that can be downloaded from:
```
http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```
or using `torchvision.datasets.CIFAR10`

### Other packages

- tqdm
- 