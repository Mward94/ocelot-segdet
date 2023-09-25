from pathlib import Path

# Grand Challenge folders were input files can be found
GC_CELL_FPATH = Path("/input/images/cell_patches/")
GC_TISSUE_FPATH = Path("/input/images/tissue_patches/")

GC_METADATA_FPATH = Path("/input/metadata.json")

# Grand Challenge output file
GC_DETECTION_OUTPUT_PATH = Path("/output/cell_classification.json")

# Sample dimensions
SAMPLE_SHAPE = (1024, 1024, 3)

# Dataloading/Model Training keys
SEG_MASK_LOGITS = 'seg_mask_logits'     # Segmentation logits
SEG_MASK_INT = 'seg_mask_int'           # Integer encoded segmentation mask
SEG_MASK_PROB = 'seg_mask_prob'         # Softmaxed segmentation mask
INPUT_IMAGE_KEY = 'input_image'         # RGB image
INPUT_MASK_PROB_KEY = 'input_mask_prob'     # Softmaxed segmentation mask (at input)
INPUT_IMAGE_MASK_KEY = 'input_image_with_mask'  # Channel-wise concatenated RGB image and seg mask
POINT_HEATMAP_KEY = 'seg_mask_point_heatmap'    # Segmentation mask representing a point heatmap
DET_POINTS_KEY = 'det_points'       # Detected point coordinates (x, y)
DET_INDICES_KEY = 'det_indices'     # Class indices of detected points
DET_SCORES_KEY = 'det_scores'       # Confidence scores per-detection
