"""
Basic data augmentations for PyTorch

Note:
- Random seed is not controlled
"""

import random
import json
import math

import numpy as np
import cv2


class ToNumpy(object):
    def __call__(self, img):
        return np.asarray(img)


class Resize(object):

    def __init__(self, output_size, interpolation=cv2.INTER_CUBIC):
        self.output_size = output_size
        self.interpolation = interpolation

    def __call__(self, img):
        # RGBA -> RGB
        if img.shape[2] == 4:
            img = img[:, :, 0:3]
        img = cv2.resize(img, dsize=self.output_size, interpolation=self.interpolation)
        return img


class RandomAugmentor(object):

    def __init__(self):
        # random = random.Random()
        pass

    def set_random_state(self, seed):
        # random = random.Random(seed)
        pass


def set_seed(compose, seed):
    """
    Method to setup seed on all transforms from torchvision.transforms.Compose
    :param compose:
    :param seed:
    :return:
    """
    for t in compose.transforms:
        if isinstance(t, RandomAugmentor):
            t.set_random_state(seed)


class RandomTransforms(RandomAugmentor):

    def __init__(self, transforms, proba=0.5):
        super(RandomTransforms, self).__init__()
        assert transforms is not None
        self.transforms = transforms
        self.proba = proba

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class RandomApply(RandomTransforms):

    def __call__(self, img):
        if self.proba < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img


class RandomOrder(RandomTransforms):

    def __call__(self, img):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            img = self.transforms[i](img)
        return img


class RandomChoice(RandomTransforms):

    def __call__(self, img):
        t = random.choice(self.transforms)
        return t(img)


class RandomFlip(RandomAugmentor):

    def __init__(self, proba=0.5, mode='h'):
        super(RandomFlip, self).__init__()
        assert mode in ['h', 'v']
        self.mode = mode
        self.proba = proba

    def __call__(self, img):
        if self.proba < random.random():
            return img
        flipCode = 1 if self.mode == 'h' else 0
        return cv2.flip(img, flipCode)


class RandomAffine(RandomAugmentor):
    """
    Random affine transform of the image w.r.t to the image center.
    Transformations involve:
    - Translation ("move" image on the x-/y-axis)
    - Rotation
    - Scaling ("zoom" in/out)
    - Shear (move one side of the image, turning a square into a trapezoid)
    """
    def __init__(self, rotation=(-90, 90), scale=None, translate=None, shear=0.0,
                 interpolation=cv2.INTER_LINEAR, border=cv2.BORDER_REPLICATE, border_value=0):
        """
        Args:
            rotation (tuple of 2 floats): rotations
            scale (tuple of 2 floats): scaling factor interval, e.g (a, b), then scale is
                randomly sampled from the range a <= scale <= b. Will keep
                original scale by default.
            translate (tuple of 2 floats): tuple of max abs fraction for horizontal
                and vertical translation. For example translate_frac=(a, b), then horizontal shift
                is randomly sampled in the range 0 < dx < img_width * a and vertical shift is
                randomly sampled in the range 0 < dy < img_height * b. Will
                not translate by default.
            shear (float): max abs shear value in degrees between 0 to 180
            interp: cv2 interpolation method
            border: cv2 border method
            border_value: cv2 border value for border=cv2.BORDER_CONSTANT
        """
        super(RandomAffine, self).__init__()
        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "Argument scale should be a tuple of two floats, e.g (a, b)"

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "Argument translate should be a tuple of two floats, e.g (a, b)"

        assert shear >= 0.0, "Argument shear should be between 0.0 and 180.0"
        self.rotation = rotation
        self.scale = scale
        self.translate = translate
        self.shear = shear
        self.interpolation = interpolation
        self.border = border
        self.border_value = border_value

    def __call__(self, img):

        if self.scale is not None:
            scale = random.uniform(self.scale[0], self.scale[1])
        else:
            scale = 1.0

        if self.translate is not None:
            max_dx = self.translate[0] * img.shape[1]
            max_dy = self.translate[1] * img.shape[0]
            dx = np.round(random.uniform(-max_dx, max_dx))
            dy = np.round(random.uniform(-max_dy, max_dy))
        else:
            dx = 0
            dy = 0

        if self.shear > 0.0:
            shear = random.uniform(-self.shear, self.shear)
            sin_shear = math.sin(math.radians(shear))
            cos_shear = math.cos(math.radians(shear))
        else:
            sin_shear = 0.0
            cos_shear = 1.0

        center = (img.shape[1::-1] * np.array((0.5, 0.5))) - 0.5
        deg = random.uniform(self.rotation[0], self.rotation[1])

        transform_matrix = cv2.getRotationMatrix2D(tuple(center), deg, scale)

        # Apply shear :
        if self.shear > 0.0:
            m00 = transform_matrix[0, 0]
            m01 = transform_matrix[0, 1]
            m10 = transform_matrix[1, 0]
            m11 = transform_matrix[1, 1]
            transform_matrix[0, 1] = m01 * cos_shear + m00 * sin_shear
            transform_matrix[1, 1] = m11 * cos_shear + m10 * sin_shear
            # Add correction term to keep the center unchanged
            tx = center[0] * (1.0 - m00) - center[1] * transform_matrix[0, 1]
            ty = -center[0] * m10 + center[1] * (1.0 - transform_matrix[1, 1])
            transform_matrix[0, 2] = tx
            transform_matrix[1, 2] = ty

        # Apply shift :
        transform_matrix[0, 2] += dx
        transform_matrix[1, 2] += dy

        ret = cv2.warpAffine(img, transform_matrix, img.shape[1::-1],
                             flags=self.interpolation,
                             borderMode=self.border,
                             borderValue=self.border_value)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret


class RandomAdd(RandomAugmentor):

    def __init__(self, proba=0.5, value=(-0, 0), per_channel=None):
        super(RandomAdd, self).__init__()
        self.proba = proba
        self.per_channel = per_channel
        self.value = value

    def __call__(self, img):
        if self.proba < random.random():
            return img
        out = img.copy()
        value = random.randint(self.value[0], self.value[1])
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

    def get_params(self, img, output_size):
        raise NotImplementedError()

    def __call__(self, img):
        if self.padding > 0:
            img = np.pad(img, self.padding, mode='edge')
        i, j, h, w = self.get_params(img, self.size)
        return img[i:i + h, j:j + w, :]


class RandomCrop(Crop, RandomAugmentor):

    def get_params(self, img, output_size):
        h, w, _ = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw


class CenterCrop(Crop):

    def get_params(self, img, output_size):
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


class AlphaLerp(RandomAugmentor):

    def __init__(self, var):
        super(AlphaLerp, self).__init__()
        if isinstance(var, (tuple, list)):
            assert len(var) == 2
            self.min_val = var[0]
            self.max_val = var[1]
        else:
            self.min_val = 0
            self.max_val = var

    def get_alpha(self):
        return random.uniform(self.min_val, self.max_val)

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

    transforms = restore_transform(transforms_json_str, custom_transforms)
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
