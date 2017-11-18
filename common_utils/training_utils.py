
import os, sys
from glob import glob
from tqdm import tqdm

import torch
from torch.autograd import Variable


def train_one_epoch(model, train_batches, criterion, optimizer, epoch, n_epochs):
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to train mode
    model.train()
    try:
        with get_tqdm(total=len(train_batches)) as pbar:
            for i, batch_data in enumerate(train_batches):

                batch_size = batch_data[0].size(0)
                batch_data = [Variable(batch_) for batch_ in batch_data]

                batch_x = [batch_ for batch_ in batch_data if len(batch_.size()) == 4]
                batch_y = [batch_ for batch_ in batch_data if len(batch_.size()) == 1]

                assert len(batch_y) == 1
                batch_y = batch_y[0]

                # compute output
                batch_y_pred = model(*batch_x)
                loss = criterion(batch_y_pred, batch_y)
                # measure accuracy and record loss

                print(batch_y_pred.size(), batch_y.size())
                print(batch_y_pred.is_cuda, batch_y.is_cuda)

                prec1 = accuracy(batch_y_pred.data, batch_y.data)
                losses.update(loss.data[0], batch_size)
                top1.update(prec1[0], batch_size)
                # compute gradient and do optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                prefix_str = "Epoch: {}/{}".format(epoch + 1, n_epochs)
                post_fix_str = 'Loss {loss.avg:.4f} | ' + \
                               'Prec@1 {top1.avg:.3f}'
                post_fix_str = post_fix_str.format(loss=losses, top1=top1)

                pbar.set_description_str(prefix_str, refresh=False)
                pbar.set_postfix_str(post_fix_str, refresh=False)
                pbar.update(1)

        return losses.avg, top1.avg
    except KeyboardInterrupt:
        return None, None


def validate(model, val_batches, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    try:
        with get_tqdm(total=len(val_batches)) as pbar:
            for i, batch_data in enumerate(val_batches):

                batch_size = batch_data[0].size(0)
                batch_data = [Variable(batch_, volatile=True) for batch_ in batch_data]
                batch_x = [batch_ for batch_ in batch_data if len(batch_.size()) == 4]
                batch_y = [batch_ for batch_ in batch_data if len(batch_.size()) == 1]
                assert len(batch_y) == 1
                batch_y = batch_y[0]
                # compute output
                batch_y_pred = model(*batch_x)
                loss = criterion(batch_y_pred, batch_y)
                # measure accuracy and record loss
                prec1 = accuracy(batch_y_pred.data, batch_y.data)
                losses.update(loss.data[0], batch_size)
                top1.update(prec1[0], batch_size)

                post_fix_str = 'Loss {loss.avg:.4f} | ' + \
                               'Prec@1 {top1.avg:.3f}'
                post_fix_str = post_fix_str.format(loss=losses, top1=top1)
                pbar.set_description_str("Validation", refresh=False)
                pbar.set_postfix_str(post_fix_str, refresh=False)
                pbar.update(1)

            return losses.avg, top1.avg
    except KeyboardInterrupt:
        return None, None


def save_checkpoint(logs_path, state):
    best_model_filenames = glob(os.path.join(logs_path, 'model_val_prec1*'))
    for fn in best_model_filenames:
        os.remove(fn)
    best_model_filename='model_val_prec1={val_prec1:.4f}.pth.tar'.format(
        val_prec1=state['val_prec1']
    )
    torch.save(state, os.path.join(logs_path, best_model_filename))


def load_checkpoint(filename, model, optimizer=None):
    print("Load checkpoint: %s" % filename)
    state = torch.load(filename)
    model.load_state_dict(state['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    return state


def write_csv_log(logs_path, line):
    csv_file = os.path.join(logs_path, 'log.csv')
    _write_log(csv_file, line)


def write_conf_log(logs_path, line):
    conf_file = os.path.join(logs_path, 'conf.log')
    _write_log(conf_file, line)


def _write_log(filename, line):
    d = 'w' if not os.path.exists(filename) else 'a'
    with open(filename, d) as w:
        w.write(line + '\n')


def verbose_optimizer(optimizer):
    msg = "\nOptimizer: %s\n" % optimizer.__class__.__name__
    msg += "Optimizer parameters: \n"
    for pg in optimizer.param_groups:
        msg += "- Param group: \n"
        for k in pg:
            if k == 'params':
                continue
            msg += "\t{}: {}\n".format(k, pg[k])
    return msg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_tqdm_kwargs(**kwargs):
    """
    Return default arguments to be used with tqdm.
    Args:
        kwargs: extra arguments to be used.
    Returns:
        dict:
    """
    default = dict(
        smoothing=0.5,
        dynamic_ncols=True,
        ascii=True,
    )
    f = kwargs.get('file', sys.stderr)
    isatty = f.isatty()
    # Jupyter notebook should be recognized as tty. Wait for
    # https://github.com/ipython/ipykernel/issues/268
    try:
        from ipykernel import iostream
        if isinstance(f, iostream.OutStream):
            isatty = True
    except ImportError:
        pass
    if isatty:
        default['mininterval'] = 0.25
    else:
        # If not a tty, don't refresh progress bar that often
        default['mininterval'] = 300
    default.update(kwargs)
    return default


def get_tqdm(**kwargs):
    """ Similar to :func:`get_tqdm_kwargs`,
    but returns the tqdm object directly. """
    return tqdm(**get_tqdm_kwargs(**kwargs))
