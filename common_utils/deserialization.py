
import json


class CustomObjectEval:
    """
    Helper class to evaluate string expression in the current context

    It is used with `restore_optimizer` for example when params defines
    "model.features.parameters" and in the current context there is object `model`
    """

    def __init__(self, globals):
        self._objects = {}
        self._globals = globals

    def __contains__(self, item_str):
        if not isinstance(item_str, str):
            return False
        try:
            obj = eval(item_str, self._globals)
            self._objects[item_str] = obj
        except NameError:
            return False
        return True

    def __getitem__(self, item_str):
        if not isinstance(item_str, str):
            return None
        return self._objects[item_str]  # can throw KeyError


def restore_object(serialized_obj, custom_objects=None, params_to_insert=None, verbose_debug=False):
    """
    Method to create optimizer class from json string or dictionary

    Example: Restore optimizer
    ```
    model = MySuperNet()
    # model has attributes : features, classifier
    custom_objects = CustomObjectEval(globals=globals())

    optimizer_json_str = '''{"Adam": {
                                "params": [{
                                    "params": {"model.features.parameters": {}},
                                    "lr": 0.0001
                                }, {"params": {"model.classifier.parameters": {}},
                                    "lr": 0.001
                                }],
                                "betas": [0.9, 0.0],
                                "eps": 1.0}
                            }'''
    optimizer = restore_optimizer(optimizer_json_str, custom_objects=custom_objects)
    ```
    Same if optimizer is defined as dict:
    ```
    optimizer_dict = {"Adam": {"params": [{
                                    "params": {"model.features.parameters": {}},
                                    "lr": 0.0001
                                }, {"params": {"model.classifier.parameters": {}},
                                    "lr": 0.001
                                }],
                                "betas": [0.9, 0.0],
                                "eps": 1.0}}
    optimizer = restore_optimizer(optimizer_dict, custom_objects=custom_objects)
    ```

    Example: Restore lr scheduler
    ```
    lr_scheduler_str = '''{
        "ExponentialLR": { "gamma": 0.77 }
    }
    '''
    # lr_scheduler_str is missing optimizer argument (this is intended as we can not know
    # in advance name of optimizer instance)
    # We insert in a posteriori
    params_to_insert = {
        'optimizer': '_opt'
    }
    custom_objects = {'_opt' : optimizer}
    lr_scheduler = restore_optimizer(optimizer_json_str,
                                     params_to_insert=params_to_insert,
                                     custom_objects=custom_objects)

    ```

    :param serialized_obj: str or dict
    :param custom_objects: dict or instance of CustomObjectEval
    :param verbose_debug:
    :return:
    """
    dict_obj = json.loads(serialized_obj) if isinstance(serialized_obj, str) else serialized_obj
    assert len(dict_obj) == 1, "serialized_obj should have single top-level key"

    if params_to_insert is not None:
        main_key = list(dict_obj.keys())[0]
        dict_obj[main_key].update(params_to_insert)
        object_json_str = json.dumps(dict_obj)
    elif not isinstance(serialized_obj, str):
        object_json_str = json.dumps(dict_obj)
    else:
        object_json_str = serialized_obj

    return json.loads(object_json_str,
                      object_hook=lambda d: object_hook(d, custom_objects, verbose_debug))


from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms


_GLOBAL_PYTORCH_OBJECTS = dict([(name, cls) for name, cls in optim.__dict__.items() if isinstance(cls, type)])
_GLOBAL_PYTORCH_OBJECTS.update([(name, cls) for name, cls in lr_scheduler.__dict__.items() if isinstance(cls, type)])
_GLOBAL_PYTORCH_OBJECTS.update(dict([(name, cls) for name, cls in transforms.__dict__.items() if isinstance(cls, type)]))


def object_hook(decoded_dict, custom_objects=None, verbose_debug=False):
    if len(decoded_dict) > 1:
        if verbose_debug:
            print("- decoded_dict: ", decoded_dict)
        # decoded_dict contains kwargs and not class name -> return
        if custom_objects is not None:
            if verbose_debug:
                print("-- check values in custom_objects")
            # replace argument's value by object if found in custom_objects
            for k, v in decoded_dict.items():
                if verbose_debug:
                    print("-- k, v: ", k, v)
                if isinstance(v, str) and v in custom_objects:
                    decoded_dict[k] = custom_objects[v]
            if verbose_debug:
                print("- Replaced decoded_dict: ", decoded_dict)
        return decoded_dict
    if verbose_debug:
        print("- type(decoded_dict): ", type(decoded_dict), decoded_dict)
    for k in decoded_dict:
        if verbose_debug:
            print("-- k: ", k)
        if custom_objects is not None and k in custom_objects:
            if verbose_debug:
                print("-- k in custom_objects, decoded_dict[k]: ", decoded_dict[k])
            return custom_objects[k](**decoded_dict[k]) if decoded_dict[k] is not None else custom_objects[k]()
        elif k in _GLOBAL_PYTORCH_OBJECTS:
            if verbose_debug:
                print("-- k in _GLOBAL_PYTORCH_OBJECTS, decoded_dict[k]: ", decoded_dict[k])
            return _GLOBAL_PYTORCH_OBJECTS[k](**decoded_dict[k]) if decoded_dict[k] is not None \
                else _GLOBAL_PYTORCH_OBJECTS[k]()
        return decoded_dict
