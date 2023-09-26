import argparse
import os

from albumentations import HorizontalFlip, GaussianBlur, GaussNoise, ColorJitter, Downscale
from torch.utils.data import DataLoader

from datasets.random_affine_crop import RandomAffineCrop
from datasets.tissue_dataset import TissueDataset
from util.torch import seed_all, get_default_device


DEFAULT_SPLIT_DIRECTORY = './splits/tissue_model' # _all_train'
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 4


def parse_args():
    parser = argparse.ArgumentParser()

    # Data source
    parser.add_argument('--data-directory', type=str,
                        help='Directory where processed data is stored', required=True)
    parser.add_argument('--split-directory', type=str,
                        help='Path to split directory.', default=DEFAULT_SPLIT_DIRECTORY)

    # Any seeding
    parser.add_argument('--seed', type=int, help='Seed for RNG.', default=None)

    # Training configuration
    parser.add_argument('--batch-size', type=int, help='Batch size', default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--num-workers', type=int, help='Number of workers', default=DEFAULT_NUM_WORKERS)

    # Output directory to store model metrics and weights
    parser.add_argument('--output-directory', type=str,
                        help='Directory to store outputs in.', required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    # Perform any seeding
    seed = args.seed
    if seed is not None:
        seed_all(seed)

    # Set up the device
    device = get_default_device()

    # Set up training dataset (with set of transforms to apply to data)
    train_transforms = [
        RandomAffineCrop(crop_size=512), HorizontalFlip(p=0.5), GaussianBlur(),
        GaussNoise(), ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        Downscale()]
    train_dataset = TissueDataset(
        args.data_directory, os.path.join(args.split_directory, 'train.txt'),
        transforms=train_transforms, samples_per_region=2, tile_size=None, output_crop_margin=None,
        scale_to_mpp=0.5, pad_class_name='Background')
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, sampler=None,
        num_workers=args.num_workers, collate_fn=None, drop_last=False)

    if os.path.isfile(os.path.join(args.split_directory, 'val.txt')):
        val_transforms = None
        val_dataset = TissueDataset(
            args.data_directory, os.path.join(args.split_directory, 'val.txt'),
            transforms=val_transforms, samples_per_region=1, tile_size=(512, 512),
            output_crop_margin=64, scale_to_mpp=0.5, pad_class_name='Background')
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, sampler=None,
            num_workers=args.num_workers, collate_fn=None, drop_last=False)
    assert False
    # TODO: Postprocessors

    # TODO: Epochs, validation behaviour/frequency

    # TODO: Optimiser

    # TODO: Scheduler

    # TODO: Loss (and whether train-only)

    # TODO: Metrics object

    # TODO: Logger for metrics

    # TODO: Train the model (helper function, pass in the things)


if __name__ == '__main__':
    main()
