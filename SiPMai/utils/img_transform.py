from torchvision import transforms
from torchvision.transforms import InterpolationMode
import random
import torchvision.transforms.functional as TF


def build_transform(mode, input_size, auto_augment, interpolation, in_channel, mean, std,
                    horizontal_flip_prob=0.5, vertical_flip_prob=0.5, rotation_range=10,
                    translate=(0.1, 0.1), scale=None, shear=None, erase_prob=0.1):
    transform_list = []
    if in_channel == 1:
        mean = mean[0]
        std = std[0]
        transform_list += [transforms.Grayscale(num_output_channels=1)]
    elif in_channel == 3:
        pass
    else:
        raise NotImplementedError

    transform_list += [
        transforms.Resize(input_size, interpolation=interpolation),
        transforms.RandomResizedCrop(input_size, interpolation=interpolation),
        transforms.RandomHorizontalFlip(horizontal_flip_prob),
        transforms.RandomVerticalFlip(vertical_flip_prob),
        transforms.RandomRotation(rotation_range),
        transforms.RandomAffine(rotation_range, translate, scale, shear),
        transforms.ToTensor()]

    if mode == "input":
        transform_list += [transforms.Normalize(mean, std),
                           transforms.RandomErasing(erase_prob)]
        if auto_augment:
            transform_list.insert(1, transforms.AutoAugment())

    elif mode == "label":
        transform_list += [lambda img: (img > 0.5).float()]
    else:
        raise NotImplementedError
    transform = transforms.Compose(transform_list)
    return transform


class PairedTransforms:
    def __init__(self, input_size, interpolation, horizontal_flip_prob=0.5, vertical_flip_prob=0.5, rotation_range=10,
                 translate=(0.1, 0.1), scale=None, shear=None):
        self.input_size = input_size
        self.interpolation = interpolation
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_range = rotation_range
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, img, mask):
        # Resize
        img = TF.resize(img, self.input_size, self.interpolation)
        mask = TF.resize(mask, self.input_size, self.interpolation)

        # RandomResizedCrop
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.input_size)
        img = TF.crop(img, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # RandomHorizontalFlip
        if random.random() < self.horizontal_flip_prob:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # RandomVerticalFlip
        if random.random() < self.vertical_flip_prob:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        # RandomRotation
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        img = TF.rotate(img, angle)
        mask = TF.rotate(mask, angle)

        # RandomAffine
        params = transforms.RandomAffine.get_params(self.rotation_range, self.translate, self.scale, self.shear,
                                                    img.size)
        img = TF.affine(img, *params, interpolation=self.interpolation)
        mask = TF.affine(mask, *params, interpolation=self.interpolation)

        return img, mask


def build_paired_transform(input_size, auto_augment, interpolation, in_channel, mean, std,
                           horizontal_flip_prob=0.5, vertical_flip_prob=0.5, rotation_range=10,
                           translate=(0.1, 0.1), scale=None, shear=None, erase_prob=0.1):
    paired_transform = PairedTransforms(input_size, interpolation, horizontal_flip_prob, vertical_flip_prob,
                                        rotation_range, translate, scale, shear)

    input_postprocess = []
    if in_channel == 1:
        input_postprocess += [transforms.Grayscale(num_output_channels=1)]
    elif in_channel == 3:
        pass
    else:
        raise NotImplementedError
    input_postprocess += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(erase_prob)
    ]

    label_postprocess = [
        transforms.ToTensor(),
        lambda tensor: (tensor > 0.5).float()
    ]

    if auto_augment:
        input_postprocess.insert(0, transforms.AutoAugment())

    def paired_transforms(img, mask):
        img, mask = paired_transform(img, mask)
        for t in input_postprocess:
            img = t(img)
        for t in label_postprocess:
            mask = t(mask)
        return img, mask

    return paired_transforms


def get_dummy_transform():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ])
    return transform


def get_dummy_paired_transform():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ])

    def paired_transforms(img, mask):
        img, mask = transform(img), transform(mask)
        return img, mask

    return paired_transforms
