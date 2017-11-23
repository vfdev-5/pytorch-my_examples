# Verification task or Siamese neural networks training

# This script  trains Siamese network on the Omniglot dataset to perform the classification task to distinguish two images of the same class or different classes.

# References:
# - [paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
# - [omniglot](https://github.com/brendenlake/omniglot)
# - [keras-oneshot](https://github.com/sorenbouma/keras-oneshot)

import os, sys
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import sigmoid
from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

from torchvision.transforms import Compose, ToTensor

sys.path.append("..")

from common_utils.imgaug import RandomAffine, RandomApply
from common_utils.dataflow import OnGPUDataLoader
from common_utils.training_utils import train_one_epoch, validate, write_csv_log, write_conf_log, verbose_optimizer, save_checkpoint
from common_utils.training_utils import accuracy

from dataflow import OmniglotDataset, SameOrDifferentPairsDataset, PairTransformedDataset
from model import SiameseNetworks


HAS_GPU = True
SEED = 12345
OMNIGLOT_REPO_PATH = 'omniglot'

# ################# Setup conf ######################

conf = {
    'nb_train_pairs': 30000,
    'nb_val_pairs': 10000,
    'nb_test_pairs': 10000,

    'weight_decay': 0.01,
    
    'lr_features': 0.00006,
    'lr_classifier': 0.00006,
    
    'n_epochs': 50,
    'batch_size': 64,
    'num_workers': 15,
    
    'gamma': 0.99,

}


# ################# Setup dataflow ######################

np.random.seed(SEED)

TRAIN_DATA_PATH = os.path.join(OMNIGLOT_REPO_PATH, 'python', 'images_background')
train_alphabets = os.listdir(TRAIN_DATA_PATH)

TEST_DATA_PATH = os.path.join(OMNIGLOT_REPO_PATH, 'python', 'images_evaluation')
test_alphabets = os.listdir(TEST_DATA_PATH)

assert len(train_alphabets) > 1 and len(test_alphabets) > 1, "%s \n %s" % (train_alphabets[0], test_alphabets[0])

train_alphabet_char_id_drawer_ids = {}
for a in train_alphabets:
    char_ids = os.listdir(os.path.join(TRAIN_DATA_PATH, a))    
    train_alphabet_char_id_drawer_ids[a] = {}
    for char_id in char_ids:
        res = os.listdir(os.path.join(TRAIN_DATA_PATH, a, char_id))
        train_alphabet_char_id_drawer_ids[a][char_id] = [_id[:-4] for _id in res]
        
        
test_alphabet_char_id_drawer_ids = {}
for a in test_alphabets:
    char_ids = os.listdir(os.path.join(TEST_DATA_PATH, a))
    test_alphabet_char_id_drawer_ids[a] = {}
    for char_id in char_ids:
        res = os.listdir(os.path.join(TEST_DATA_PATH, a, char_id))
        test_alphabet_char_id_drawer_ids[a][char_id] = [_id[:-4] for _id in res]


# Sample 12 drawers out of 20
all_drawers_ids = np.arange(20) 
train_drawers_ids = np.random.choice(all_drawers_ids, size=12, replace=False)
# Sample 4 drawers out of remaining 8
val_drawers_ids = np.random.choice(list(set(all_drawers_ids) - set(train_drawers_ids)), size=4, replace=False)
test_drawers_ids = np.array(list(set(all_drawers_ids) - set(val_drawers_ids) - set(train_drawers_ids)))


def create_str_drawers_ids(drawers_ids):
    return ["_{0:0>2}".format(_id) for _id in drawers_ids]

train_drawers_ids = create_str_drawers_ids(train_drawers_ids)
val_drawers_ids = create_str_drawers_ids(val_drawers_ids)
test_drawers_ids = create_str_drawers_ids(test_drawers_ids)

train_ds = OmniglotDataset("Train", data_path=TRAIN_DATA_PATH, 
                           alphabet_char_id_drawers_ids=train_alphabet_char_id_drawer_ids, 
                           drawers_ids=train_drawers_ids)

val_ds = OmniglotDataset("Test", data_path=TEST_DATA_PATH, 
                         alphabet_char_id_drawers_ids=test_alphabet_char_id_drawer_ids, 
                         drawers_ids=val_drawers_ids)

test_ds = OmniglotDataset("Test", data_path=TEST_DATA_PATH, 
                          alphabet_char_id_drawers_ids=test_alphabet_char_id_drawer_ids, 
                          drawers_ids=test_drawers_ids)

train_pairs = SameOrDifferentPairsDataset(train_ds, nb_pairs=int(30e3))
val_pairs = SameOrDifferentPairsDataset(val_ds, nb_pairs=int(10e3))
test_pairs = SameOrDifferentPairsDataset(test_ds, nb_pairs=int(10e3))

train_data_aug = Compose([
    RandomApply(
        RandomAffine(rotation=(-10, 10), scale=(0.8, 1.2), translate=(-0.05, 0.05)),
        proba=0.5
    ),
    ToTensor()
])

test_data_aug = Compose([
    ToTensor()
])

y_transform = lambda y: torch.FloatTensor([y])

train_aug_pairs = PairTransformedDataset(train_pairs, x_transforms=train_data_aug, y_transforms=y_transform)
val_aug_pairs = PairTransformedDataset(val_pairs, x_transforms=test_data_aug, y_transforms=y_transform)
test_aug_pairs = PairTransformedDataset(test_pairs, x_transforms=test_data_aug, y_transforms=y_transform)

_DataLoader = OnGPUDataLoader if HAS_GPU and torch.cuda.is_available() else DataLoader

train_batches = _DataLoader(train_aug_pairs, batch_size=conf['batch_size'], 
                            shuffle=True, num_workers=5, 
                            drop_last=True)

val_batches = _DataLoader(val_aug_pairs, batch_size=conf['batch_size'],
                          shuffle=True, num_workers=conf['num_workers'],
                          pin_memory=True, drop_last=True)

test_batches = _DataLoader(test_aug_pairs, batch_size=conf['batch_size'], 
                           shuffle=False, num_workers=conf['num_workers'],                   
                           pin_memory=True, drop_last=False)


# ################# Setup model and optimization algorithm ######################

siamese_net = SiameseNetworks(input_shape=(105, 105, 1))
if HAS_GPU and torch.cuda.is_available():
    siamese_net = siamese_net.cuda()


def accuracy_logits(y_logits, y_true):
    y_pred = sigmoid(y_logits).data
    return accuracy(y_pred, y_true)

criterion = BCEWithLogitsLoss()
if HAS_GPU and torch.cuda.is_available():
    criterion = criterion.cuda()

optimizer = Adam([{
    'params': siamese_net.net.features.parameters(),
    'lr': conf['lr_features'],    
}, {
    'params': siamese_net.classifier.parameters(),
    'lr': conf['lr_classifier']
}], 
    weight_decay=conf['weight_decay']
)

# lr <- lr_init * gamma ** epoch
scheduler = ExponentialLR(optimizer, gamma=conf['gamma'])
onplateau_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

now = datetime.now()
logs_path = os.path.join('logs', 'seamese_networks_verification_task_%s' % (now.strftime("%Y%m%d_%H%M")))
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    
    
write_conf_log(logs_path, "{}".format(conf))
write_conf_log(logs_path, verbose_optimizer(optimizer))

write_csv_log(logs_path, "epoch,train_loss,train_acc,val_loss,val_acc")

best_acc = 0.0
for epoch in range(conf['n_epochs']):
    scheduler.step()
    # Verbose learning rates:
    print(verbose_optimizer(optimizer))

    # train for one epoch
    ret = train_one_epoch(siamese_net, train_batches, 
                          criterion, optimizer,                                               
                          epoch, conf['n_epochs'], avg_metrics=[accuracy_logits,])
    if ret is None:
        break
    train_loss, train_acc = ret

    # evaluate on validation set
    ret = validate(siamese_net, val_batches, criterion, avg_metrics=[accuracy_logits, ])
    if ret is None:
        break
    val_loss, val_acc = ret
    
    onplateau_scheduler.step(val_loss)

    # Write a csv log file
    write_csv_log(logs_path, "%i,%f,%f,%f,%f" % (epoch, train_loss, train_acc, val_loss, val_acc))
    
    # remember best accuracy and save checkpoint
    if val_acc > best_acc:
        best_prec1 = max(val_acc, best_acc)
        save_checkpoint(logs_path, 'val_acc', 
                        {'epoch': epoch + 1,
                         'state_dict': siamese_net.state_dict(),
                         'val_acc': val_acc,           
                         'optimizer': optimizer.state_dict()})
