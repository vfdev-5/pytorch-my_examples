
import os, sys
from glob import glob
from tqdm import tqdm

import torch
from torch.autograd import Variable


def train_one_epoch(model, train_batches, criterion, optimizer, epoch, n_epochs, avg_metrics=None):
    """
    :param model: class derived from nn.Module
    :param train_batches: instance of DataLoader
    :param criterion: loss function, callable with signature loss = criterion(batch_y_pred, batch_y)
    :param optimizer:
    :param epoch:
    :param n_epochs:
    :param avg_metrics: list of metrics functions, e.g. [metric_fn1, metric_fn2, ...]
        for example, accuracy(batch_y_pred_tensor, batch_y_true_tensor) -> value
    :return: list of averages, [loss, ] or [loss, metric1, metric2] if metrics is defined
    """

    # Loss
    average_meters = [AverageMeter()]

    if avg_metrics is not None:
        average_meters.extend([AverageMeter() for _ in avg_metrics])

    # switch to train mode
    model.train()
    try:
        with get_tqdm(total=len(train_batches)) as pbar:
            for i, (batch_x, batch_y) in enumerate(train_batches):

                assert torch.is_tensor(batch_y)
                batch_size = batch_y.size(0)

                if isinstance(batch_x, list):
                    batch_x = [Variable(batch_, requires_grad=True) for batch_ in batch_x]
                else:
                    batch_x = [Variable(batch_x, requires_grad=True)]
                batch_y = Variable(batch_y)

                # compute output and measure loss
                batch_y_pred = model(*batch_x)
                loss = criterion(batch_y_pred, batch_y)
                average_meters[0].update(loss.data[0], batch_size)

                prefix_str = "Epoch: {}/{}".format(epoch + 1, n_epochs)
                pbar.set_description_str(prefix_str, refresh=False)
                post_fix_str = "Loss {loss.avg:.4f}".format(loss=average_meters[0])

                # measure metrics
                if avg_metrics is not None:
                    for _fn, av_meter in zip(avg_metrics, average_meters[1:]):
                        v = _fn(batch_y_pred.data, batch_y.data)
                        av_meter.update(v, batch_size)
                        post_fix_str += " | {name} {av_meter.avg:.3f}".format(name=_fn.__name__, av_meter=av_meter)

                pbar.set_postfix_str(post_fix_str, refresh=False)
                pbar.update(1)

                # compute gradient and do optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return [m.avg for m in average_meters]
    except KeyboardInterrupt:
        return None


def validate(model, val_batches, criterion, avg_metrics=None, full_data_metrics=None):
    """
    :param model:
    :param val_batches:
    :param criterion:
    :param avg_metrics:
    :param full_data_metrics:
    :return:
    """

    # Loss
    average_meters = [AverageMeter()]

    if avg_metrics is not None:
        average_meters.extend([AverageMeter() for _ in avg_metrics])

    y_true_full = []
    y_pred_full = []

    # switch to evaluate mode
    model.eval()
    try:
        with get_tqdm(total=len(val_batches)) as pbar:
            for i, (batch_x, batch_y) in enumerate(val_batches):

                assert torch.is_tensor(batch_y)
                batch_size = batch_y.size(0)

                if isinstance(batch_x, list):
                    batch_x = [Variable(batch_, volatile=True) for batch_ in batch_x]
                else:
                    batch_x = [Variable(batch_x, volatile=True)]
                batch_y = Variable(batch_y, volatile=True)

                # compute output and measure loss
                batch_y_pred = model(*batch_x)
                loss = criterion(batch_y_pred, batch_y)
                average_meters[0].update(loss.data[0], batch_size)

                if full_data_metrics is not None:
                    _batch_y = batch_y.data
                    if _batch_y.cuda:
                        _batch_y = _batch_y.cpu()
                    y_true_full.append(_batch_y.numpy())
                    _batch_y_pred = batch_y_pred.data
                    if _batch_y_pred.cuda:
                        _batch_y_pred = batch_y_pred.cpu()
                    y_pred_full.append(_batch_y_pred.numpy())

                # measure average metrics
                post_fix_str = "Loss {loss.avg:.4f}".format(loss=average_meters[0])
                # measure metrics
                if avg_metrics is not None:
                    for _fn, av_meter in zip(avg_metrics, average_meters[1:]):
                        v = _fn(batch_y_pred.data, batch_y.data)
                        av_meter.update(v, batch_size)
                        post_fix_str += " | {name} {av_meter.avg:.3f}".format(name=_fn.__name__, av_meter=av_meter)

                pbar.set_postfix_str(post_fix_str, refresh=False)
                pbar.update(1)

            if full_data_metrics is not None:
                res = []
                for _fn in full_data_metrics:
                    res.append(_fn(y_true_full, y_pred_full))
                return [m.avg for m in average_meters], res
            else:
                return [m.avg for m in average_meters]
    except KeyboardInterrupt:
        return None


def save_checkpoint(logs_path, val_metric_name, state):
    best_model_filenames = glob(os.path.join(logs_path, 'model_%s*' % val_metric_name))
    for fn in best_model_filenames:
        os.remove(fn)
    best_model_filename = 'model_%s={val_metric_name:.4f}.pth.tar' % val_metric_name
    best_model_filename = best_model_filename.format(
        val_metric_name=state[val_metric_name]
    )
    torch.save(state, os.path.join(logs_path, best_model_filename))


def load_checkpoint(filename, model, optimizer=None):
    print("Load checkpoint: %s" % filename)
    state = torch.load(filename)
    model.load_state_dict(state['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])


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

    if output.size(1) > 1:
        _, pred = output.topk(maxk)
    else:
        pred = torch.round(output)

    if len(target.size()) == 1:
        target = target.view(-1, 1)
    
    if pred.type() != target.type():
        target = target.type_as(pred)

    correct = pred.eq(target.expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(correct_k * (1.0 / batch_size))
    return res if len(topk) > 1 else res[0]


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
