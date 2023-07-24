from torchvision import transforms

def build_transform(split, input_size, auto_augment, interpolation, mean, std,
                    horizontal_flip_prob=0.5, vertical_flip_prob=0.5, rotation_range=10,
                    translate=(0.1, 0.1), scale=None, shear=None, erase_prob=0.1):

    if split == "train":
        transform_list = [
            transforms.RandomResizedCrop(input_size, interpolation=interpolation),
            transforms.RandomHorizontalFlip(horizontal_flip_prob),
            transforms.RandomVerticalFlip(vertical_flip_prob),
            transforms.RandomRotation(rotation_range),
            transforms.RandomAffine(rotation_range, translate, scale, shear),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(erase_prob),
        ]
        if auto_augment:
            transform_list.insert(1, transforms.AutoAugment())
        transform = transforms.Compose(transform_list)
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(input_size[0] + 32),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transform
