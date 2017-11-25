In this section we presents the paper ["Siamese Neural Networks for One-shot Image Recognition"](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) coded with PyTorch framework. 

## TL;DR

- Dataset is Omniglot: 1623 characters, 20 number of samples per character (*transposed MNIST*)
- Siamese network as two sharing weights CNNs predicts if two images are of the same class or not
- Verification task = training of the siamese network
- One-shot learning = may only observe a single example of each possible class before making a prediction about a test instance
- N-way evaluation: 
    - test image to classify as one of N classes 
    - support set: N (unseen) images, one representative for each class

```
test image ------[CNN]---- features -----\
                                          \
train image 1 ---[CNN]---- features ---[is same?] ---- Proba class 1
train image 2 ---[CNN]---- features ---[is same?] ---- Proba class 2  
    ...                                  
train image N ---[CNN]---- features ---[is same?] ---- Proba class N
```

### Results:

One-shot learning evaluation is done as described in the paper on 10 test alphabets 2 trials by 20-way on each alphabet

| Training | | One-shot learning |   
--- | --- | --- | ---
nb of pairs | nb epochs | val accuracy | mean accuracy  | mean accuracy@3 
    100k    |    50     |   0.8728     | 0.3725         | 0.67125
    


## Details 

In this tutorial:
> we explore a method for learning *siamese neural networks* which employ a unique structure to naturally rank 
similarity between inputs. Once a network has been tuned, we can then capitalize on powerful discriminative 
features to generalize the predictive power of the network not just to new data, but to entirely new classes 
from unknown distributions. 

### Model

Two convolutional neural networks with shared weights are used to extract features from two images and predict whether 
they are from the same class or from different classes:

```
train image 1 ------[CNN]---- features --------\
                                         [L2 distance] ---- [binary classifier] ---- probability    
train image 2 ------[CNN]---- features --------/ 
``` 


### Omniglot dataflow

In the notebook `Siamese_Networks__2_Omniglot_dataflow.ipynb` we present the dataset and setup dataflow 
for the training task.

To train siamese networks we need to setup a dataflow that provides image pairs of the same class and of different 
classes. 

Following the paper we apply random geometrical data augmentations:
- small rotations
- translations
- scale

### Training strategy

Training and validation of the model is done on image pairs. Dataset split is done according to the paper.

However, contrarily to the paper, we do not produce augmented datasets of 270000, 810000, and 1350000 
effective examples. We train on ~100k pairs produced with random geometrical transformations.

 
### One-shot learning evaluation

We follow paper's strategy and perform an N-way evaluation on one alphabet. Idea is to choose one test alphabet, 
select N characters drawn by one drawer (N test images) and N characters drawn by another drawer (support set). 
Next we use the trained model to compare test image with the support set, compute probabilities and select the most 
similar image from support set and deduce the class:   
```
test image ------[CNN]---- features -----\
                                          \
train image 1 ---[CNN]---- features ---[is same?] ---- Proba class 1
train image 2 ---[CNN]---- features ---[is same?] ---- Proba class 2  
    ...                                  
train image N ---[CNN]---- features ---[is same?] ---- Proba class N
```


References:
- [paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [omniglot](https://github.com/brendenlake/omniglot)
- [keras-oneshot](https://github.com/sorenbouma/keras-oneshot)