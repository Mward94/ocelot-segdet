"""Training script for the cell detection model.

Training details defined in code (not configurable by CLA):
- Gaussian heatmap ground truth sigma: 1.14um (~8um within 7 stds)
- Trained with a single channel 'cancer area' heatmap concatenated with the RGB input image
- Postprocessors: Seg Mask to Point Heatmap, Gaussian Modulation (5.714px ~= 1.14um @ 0.2MPP),
                  Blob Detection, PointNMS (15px distance threshold)
- Training augmentations: RandomAffineCrop, HorizontalFlip, GaussianBlur, GaussNoise,
                          ColorJitter, Downscale
- Optimiser: AdamW
- LR scheduler: Linearly decays LR to 0
- Loss: MSE Loss on segmentation mask.
- Tiles the validation dataset with 512x512px tiles
"""
import argparse
import os

from albumentations import HorizontalFlip, GaussianBlur, GaussNoise, ColorJitter, Downscale
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets.cell_dataset import CellDataset
from datasets.random_affine_crop import RandomAffineCrop
from networks.criteria.point_heatmap_mse_loss import PointHeatmapMSELoss
from networks.metrics.point_det_metrics import PointDetMetrics
from networks.postprocessors.gaussian_modulation import GaussianModulation
from networks.postprocessors.point_heatmap_blob_detector import PointHeatmapBlobDetector
from networks.postprocessors.point_nms import PointNMS
from networks.postprocessors.seg_mask_to_point_heatmap import SegMaskToPointHeatmap
from networks.schedulers.polynomial_lr_decay import PolynomialLRDecay
from networks.segformer import SegFormer
from util.constants import CELL_CLASSES
from util.csv_logger import CSVLogger
from util.helpers import create_directory
from util.torch import seed_all, get_default_device, detection_collate_fn
from util.training import train_model

# Training related
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 4
DEFAULT_EPOCHS = 250
DEFAULT_LR = 0.00006
DEFAULT_VAL_FREQ = 2
DEFAULT_VAL_BEHAVIOUR = 'reconstruct'

# Data related
DEFAULT_SPLIT_DIRECTORY = './splits/cell_model' # _all_train'
DEFAULT_MPP = 0.2       # MPP to scale data to

# Output crop margins
DEFAULT_TRAIN_OCM = 0
DEFAULT_EVAL_OCM = 128

# Model related
DEFAULT_SEGFORMER_SIZE = 'b2'
DEFAULT_SEGFORMER_PRETRAINED = True

# Gaussian ground truth
GAUSSIAN_GT_SIGMA = 1.14
GAUSSIAN_GT_SIGMA_UNITS = 'microns'

# Postprocessors
GAUSSIAN_MOD_SIGMA = 5.714      # [(8/7) / 0.2MPP]


def parse_args():
    parser = argparse.ArgumentParser()

    # Data source
    parser.add_argument('--data-directory', type=str,
                        help='Directory where processed data is stored', required=True)
    parser.add_argument('--split-directory', type=str,
                        help='Path to split directory.', default=DEFAULT_SPLIT_DIRECTORY)
    parser.add_argument('--cancer-area-heatmap-directory', type=str,
                        help='Directory where predicted cancer area heatmaps are stored',
                        required=True)

    # Any seeding
    parser.add_argument('--seed', type=int, help='Seed for RNG. <0 = no seed',
                        default=DEFAULT_SEED)

    # Data configuration
    parser.add_argument('--mpp', type=float,
                        help='MPP to scale data to. <=0 = no scaling.', default=DEFAULT_MPP)

    # Training configuration
    parser.add_argument('--batch-size', type=int, help='Batch size',
                        default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--num-workers', type=int, help='Number of workers',
                        default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--epochs', type=int, help='Number of epochs to train for',
                        default=DEFAULT_EPOCHS)
    parser.add_argument('--initial-lr', type=float, help='Initial learning rate',
                        default=DEFAULT_LR)
    parser.add_argument('--validation-frequency', type=int,
                        help='Frequency (in epochs) to perform evaluation over validation set '
                             '(if one exists). Setting > 1 can speed up training',
                        default=DEFAULT_VAL_FREQ)
    parser.add_argument('--validation-behaviour', type=str, choices=['single', 'reconstruct'],
                        help='Validation set metric computation behaviour used when tiling. If '
                             '\'reconstruct\', all tiles will first be reassembled before metric '
                             'computation (this is more accurate, however slower). If \'single\', '
                             'metrics will be computed per-tile. If not tiling, this should be set '
                             'to \'single\'.',
                        default=DEFAULT_VAL_BEHAVIOUR)

    # Output crop margins
    parser.add_argument('--train-ocm', type=int,
                        help='Output crop margin when training', default=DEFAULT_TRAIN_OCM)
    parser.add_argument('--eval-ocm', type=int,
                        help='Output crop margin when evaluating', default=DEFAULT_EVAL_OCM)

    # Model configuration
    parser.add_argument('--segformer-size', type=str, help='Size of SegFormer model',
                        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'], default=DEFAULT_SEGFORMER_SIZE)
    parser.add_argument('--segformer-pretrained', type=argparse.BooleanOptionalAction,
                        help='Use ImageNet pre-trained weights for SegFormer',
                        default=DEFAULT_SEGFORMER_PRETRAINED)

    # Other options
    parser.add_argument('--compute-train-metrics', action=argparse.BooleanOptionalAction,
                        help='Compute metrics on training set', default=True)
    parser.add_argument('--compute-val-loss', action=argparse.BooleanOptionalAction,
                        help='Compute loss on validation set', default=True)

    # Output directory to store model metrics and weights
    parser.add_argument('--output-directory', type=str,
                        help='Directory to store outputs in.', required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    # Perform any seeding
    seed = args.seed
    if seed >= 0:
        seed_all(seed)

    # Set up MPP to use
    mpp = args.mpp
    if mpp <= 0:
        mpp = None

    # Set up the device
    device = get_default_device()

    # Set up training dataset (with set of transforms to apply to data)
    train_transforms = [
        RandomAffineCrop(crop_size=512), HorizontalFlip(p=0.5), GaussianBlur(),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        Downscale()]
    train_dataset = CellDataset(
        args.data_directory, os.path.join(args.split_directory, 'train.txt'),
        transforms=train_transforms, samples_per_region=1, tile_size=None,
        output_crop_margin=args.train_ocm, scale_to_mpp=mpp, gaussian_sigma=GAUSSIAN_GT_SIGMA,
        gaussian_sigma_units=GAUSSIAN_GT_SIGMA_UNITS,
        seg_mask_dir=args.cancer_area_heatmap_directory)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, sampler=None,
        num_workers=args.num_workers, collate_fn=detection_collate_fn, drop_last=False)

    if os.path.isfile(os.path.join(args.split_directory, 'val.txt')):
        val_transforms = None
        val_dataset = CellDataset(
            args.data_directory, os.path.join(args.split_directory, 'val.txt'),
            transforms=val_transforms, samples_per_region=1, tile_size=(512, 512),
            output_crop_margin=args.eval_ocm, scale_to_mpp=mpp, gaussian_sigma=GAUSSIAN_GT_SIGMA,
            gaussian_sigma_units=GAUSSIAN_GT_SIGMA_UNITS,
            seg_mask_dir=args.cancer_area_heatmap_directory)
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, sampler=None,
            num_workers=args.num_workers, collate_fn=detection_collate_fn, drop_last=False)
    else:
        val_dataloader = None

    # Set up model
    model = SegFormer(num_classes=len(CELL_CLASSES), size=args.segformer_size,
                      pretrained=args.segformer_pretrained,
                      train_output_crop_margin=args.train_ocm,
                      eval_output_crop_margin=args.eval_ocm, input_with_mask=True, mask_channels=1)
    model.train()

    # Set up postprocessors
    postprocessors = [SegMaskToPointHeatmap(), GaussianModulation(sigma=GAUSSIAN_MOD_SIGMA),
                      PointHeatmapBlobDetector(ignore_index=CELL_CLASSES.index('Background')),
                      PointNMS(euc_dist_threshold=15)]

    # Set up parameters for model training
    num_epochs = args.epochs
    val_freq = 0 if val_dataloader is None else args.validation_frequency

    # Set up optimiser
    optimiser = AdamW(params=model.parameters(), lr=args.initial_lr)

    # Set up scheduler
    scheduler = PolynomialLRDecay(optimiser=optimiser, max_epoch=num_epochs, power=1.0, min_lr=0)

    # Set up object for computing loss
    criterion_object = PointHeatmapMSELoss(ignore_class='Background', class_list=CELL_CLASSES)

    # Set up object for metric computation
    metrics_object = PointDetMetrics(
        class_list=CELL_CLASSES, micron_thresholds=[3], compute_conf_mat_info=True,
        ignore_class='Background')

    # Set up output directory and create logger for metrics
    create_directory(args.output_directory)
    print(f'Writing all data to: {args.output_directory}')
    train_log = CSVLogger(os.path.join(args.output_directory, 'train_log.csv'), overwrite=True)
    if val_dataloader is not None:
        val_log = CSVLogger(os.path.join(args.output_directory, 'val_log.csv'), overwrite=True)
    else:
        val_log = None

    # Train the model
    train_model(model, optimiser, train_dataloader, val_dataloader, device, num_epochs,
                args.output_directory, criterion_object, scheduler, metrics_object, train_log,
                val_log, criterion_train_only=not args.compute_val_loss,
                metrics_val_only=not args.compute_train_metrics, validation_frequency=val_freq,
                validation_behaviour=args.validation_behaviour, postprocessors=postprocessors)


if __name__ == '__main__':
    main()
