In this section we presents the paper ["Siamese Neural Networks for One-shot Image Recognition"](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) coded with PyTorch framework. 

## TL;DR

- Dataset is Omniglot: 1623 characters, 20 number of samples per character (*transposed MNIST*)
- Siamese network as two sharing weights CNNs predicts if two images are of the same class or not
- Verification task = training of the siamese network
- One-shot learning = may only observe a single example of each possible class before making a prediction about a test instance
- N-way evaluation ...

```
test image ------[CNN]---- features ----\
                                         \
train image 1 ---[CNN]---- features ---[is same?] ---- Proba class 1
train image 2 ---[CNN]---- features ---[is same?] ---- Proba class 2  
    ...                                  
train image N ---[CNN]---- features ---[is same?] ---- Proba class N

```


## Details 

In this tutorial:
> we explore a method for learning *siamese neural networks* which employ a unique structure to naturally rank similarity between inputs. Once a network has been tuned, we can then capitalize on powerful discriminative features to generalize the predictive power of the network not just to new data, but to entirely new classes from unknown distributions. 

### Model



### Omniglot dataflow

In the notebook `Siamese_Networks__2_Omniglot_dataflow.ipynb` we present the dataset and setup dataflow for the verification task.


### Training strategy

Two steps:

#### Verification task

Verification task is a step in the training strategy to train the siamese network before one-shot learning evaluation.

#### One-shot learning evaluation


...


References:
- [paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [omniglot](https://github.com/brendenlake/omniglot)
- [keras-oneshot](https://github.com/sorenbouma/keras-oneshot)