"""
Basic data augmentations for PyTorch

Note:
- Random seed is not controlled
- Methods of Transforms serialization/deserialization
"""

import json

import numpy as np
import cv2


class ToNumpy(object):
    def __call__(self, img):
        return np.asarray(img)


class RandomOrder(object):
    def __init__(self, transforms):
        assert transforms is not None
        self.transforms = transforms

    def __call__(self, img):
        order = np.random.permutation(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class RandomChoice(object):

    def __init__(self, transforms):
        assert transforms is not None
        self.transforms = transforms

    def __call__(self, img):
        c = np.random.choice(len(self.transforms), 1)[0]
        return self.transforms[c](img)


class RandomFlip(object):

    def __init__(self, proba=0.5, mode='h'):
        assert mode in ['h', 'v']
        self.mode = mode
        self.proba = proba

    def __call__(self, img):
        if self.proba > np.random.rand():
            return img
        flipCode = 1 if self.mode == 'h' else 0
        return cv2.flip(img, flipCode)


class RandomAffine(object):

    def __init__(self, rotation=(-90, 90), scale=(0.85, 1.15), translate=(0.2, 0.2)):
        self.rotation = rotation
        self.scale = scale
        self.translate = translate

    def __call__(self, img):

        scale = np.random.uniform(self.scale[0], self.scale[1])
        deg = np.random.uniform(self.rotation[0], self.rotation[1])

        max_dx = self.translate[0] * img.shape[1]
        max_dy = self.translate[1] * img.shape[0]
        dx = np.round(np.random.uniform(-max_dx, max_dx))
        dy = np.round(np.random.uniform(-max_dy, max_dy))

        center = (img.shape[1::-1] * np.array((0.5, 0.5))) - 0.5
        transform_matrix = cv2.getRotationMatrix2D(tuple(center), deg, scale)

        # Apply shift :
        transform_matrix[0, 2] += dx
        transform_matrix[1, 2] += dy

        ret = cv2.warpAffine(img, transform_matrix, img.shape[1::-1],
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret


class RandomAdd(object):

    def __init__(self, proba=0.5, value=(-0, 0), per_channel=None):
        self.proba = proba
        self.per_channel = per_channel
        self.value = value

    def __call__(self, img):
        if self.proba > np.random.rand():
            return img
        out = img.copy()
        value = np.random.randint(self.value[0], self.value[1])
        if self.per_channel is not None and \
                        self.per_channel in list(range(img.shape[-1])):
            out[:, :, self.per_channel] = np.clip(out[:, :, self.per_channel] + value, 0, 255)
        else:
            out[:, :, :] = np.clip(out[:, :, :] + value, 0, 255)
        return out


class Crop(object):

    def __init__(self, size, padding=0):
        assert len(size) == 2
        self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        raise NotImplementedError()

    def __call__(self, img):
        if self.padding > 0:
            img = np.pad(img, self.padding, mode='edge')
        i, j, h, w = self.get_params(img, self.size)
        return img[i:i + h, j:j + w, :]


class RandomCrop(Crop):

    @staticmethod
    def get_params(img, output_size):
        h, w, _ = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = np.random.randint(0, h - th)
        j = np.random.randint(0, w - tw)
        return i, j, th, tw


class CenterCrop(Crop):

    @staticmethod
    def get_params(img, output_size):
        h, w, _ = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = (h - th) // 2
        j = (w - tw) // 2
        return i, j, th, tw


# #### Next code is adapted from here :
# #### https://github.com/pytorch/vision/blob/659c854c6971ecc5b94dca3f4459ef2b7e42fb70/torchvision/transforms.py

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class AlphaLerp(object):

    def __init__(self, var):
        if isinstance(var, (tuple, list)):
            assert len(var) == 2
            self.min_val = var[0]
            self.max_val = var[1]
        else:
            self.min_val = 0
            self.max_val = var

    def get_alpha(self):
        return np.random.uniform(self.min_val, self.max_val)

    def get_end_image(self, img):
        raise NotImplementedError

    def __call__(self, img):
        return img.lerp(self.get_end_image(img), self.get_alpha())


class Saturation(AlphaLerp):

    def __init__(self, var):
        super(Saturation, self).__init__(var)

    def get_end_image(self, img):
        return Grayscale()(img)


class Brightness(AlphaLerp):

    def __init__(self, var):
        super(Brightness, self).__init__(var)

    def get_end_image(self, img):
        return img.new().resize_as_(img).zero_()


class Contrast(AlphaLerp):

    def __init__(self, var):
        super(Contrast, self).__init__(var)

    def get_end_image(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        return gs


class ColorJitter(RandomOrder):

    def __init__(self, brightness=None, contrast=None, saturation=None):
        """
        :param brightness: int or tuple: (min, max) with min < max in [0.0, 1.0]
        :param contrast: int or tuple: (min, max) with min < max in [0.0, 1.0]
        :param saturation: int or tuple: (min, max) with min < max in [0.0, 1.0]
        """
        assert brightness or contrast or saturation
        transforms = []
        if brightness is not None:
            transforms.append(Brightness(brightness))
        if contrast is not None:
            transforms.append(Contrast(contrast))
        if saturation is not None:
            transforms.append(Saturation(saturation))
        super(ColorJitter, self).__init__(transforms)

# #### END


def to_json(transforms):
    serialization = lambda o: {o.__class__.__name__: o.__dict__}
    return json.dumps(transforms, default=serialization)


def restore_transform(transforms_json_str, custom_transforms=None):
    """
    Method to create a Transform class from json string

    Example:
    ```
    transforms_json_str = '''{"Compose":
                            {"transforms": [{"RandomCrop": {"padding": 0, "size": [299, 299]}},
                                            {"RandomChoice": {"transforms": [
                                                {"RandomAffine": {"translate": [0.05, 0.05],
                                                                    "rotation": [-60, 60],
                                                                    "scale": [0.95, 1.05]}},
                                                {"RandomFlip": {"mode": "h", "proba": 0.5}},
                                                {"RandomFlip": {"mode": "v", "proba": 0.5}}]}},
                                                {"ToTensor": {}},
                                                {"Normalize": {"std": [0.229, 0.224, 0.225],
                                                                "mean": [0.485, 0.456, 0.406]}
                                                }]
                                            }
                            }'''
    custom_transforms = {
        "RandomCrop": RandomCrop,
        "RandomChoice": RandomChoice,
        "RandomAffine": RandomAffine,
        "RandomFlip": RandomFlip
    }
    ```
    :param transforms_json_str:
    :param custom_transforms:
    :return:
    """
    return json.loads(transforms_json_str,
                      object_hook=lambda d: object_hook(d, custom_transforms))

from torchvision import transforms

_GLOBAL_TRANSFORMS = dict([(name, cls) for name, cls in transforms.__dict__.items() if isinstance(cls, type)])


def object_hook(decoded_dict, custom_transforms=None):
    if len(decoded_dict) > 1:
        # decoded_dict contains kwargs and not class name
        return decoded_dict
    for k in decoded_dict:
        if custom_transforms is not None and k in custom_transforms:
            return custom_transforms[k](**decoded_dict[k]) if decoded_dict[k] is not None else custom_transforms[k]()
        elif k in _GLOBAL_TRANSFORMS:
            return _GLOBAL_TRANSFORMS[k](**decoded_dict[k]) if decoded_dict[k] is not None else _GLOBAL_TRANSFORMS[k]()
        return decoded_dict
