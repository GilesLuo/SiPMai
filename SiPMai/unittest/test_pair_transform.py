import unittest
import torch
from SiPMai.utils.img_transform import build_paired_transform, PairedTransforms

# Assuming the improved code is in a module named "paired_transform_module"
# from paired_transform_module import PairedTransforms, build_paired_transform

class TestPairedTransforms(unittest.TestCase):

    def setUp(self):
        # Create a dummy image-mask pair
        self.img = torch.randn(3, 64, 64)  # A random 3-channel image
        self.mask = torch.randn(1, 64, 64)  # A random mask

        # Set fixed parameters for the transform
        self.input_size = (32, 32)
        self.interpolation = InterpolationMode.BILINEAR

    def test_paired_transforms_consistency(self):
        transform = PairedTransforms(self.input_size, self.interpolation, seed=42)
        transformed_img, transformed_mask = transform(self.img, self.mask)

        # Check if the dimensions are consistent
        self.assertEqual(transformed_img.shape, transformed_mask.shape)

        # Add more checks to validate if the transformations applied on img and mask are consistent

    def test_postprocess(self):
        paired_transform = build_paired_transform(self.input_size, False, self.interpolation, 3, [0.5, 0.5, 0.5],
                                                  [0.1, 0.1, 0.1])
        transformed_img, transformed_mask = paired_transform(self.img, self.mask)

        # Check normalization (assuming normalization mean and std are known)
        self.assertAlmostEqual(transformed_img.mean().item(), 0.5, places=1)
        self.assertAlmostEqual(transformed_img.std().item(), 0.1, places=1)

        # Add more post-process checks

    def test_auto_augment(self):
        paired_transform = build_paired_transform(self.input_size, True, self.interpolation, 3, [0.5, 0.5, 0.5],
                                                  [0.1, 0.1, 0.1])
        transformed_img, transformed_mask = paired_transform(self.img, self.mask)

        # This is a bit tricky since the effect of AutoAugment is not easily verifiable.
        # However, you can verify if the returned tensors are different from the input tensors.
        self.assertFalse(torch.equal(self.img, transformed_img))
        self.assertFalse(torch.equal(self.mask, transformed_mask))

    def test_edge_cases(self):
        # Test with edge parameters and validate the behavior

        pass

    def test_parameter_checks(self):
        # Test with unsupported parameters and check for expected exceptions

        with self.assertRaises(NotImplementedError):
            build_paired_transform(self.input_size, False, self.interpolation, 4, [0.5, 0.5, 0.5], [0.1, 0.1, 0.1])


# Add more tests as required

# Run the tests
if __name__ == "__main__":
    unittest.main()
