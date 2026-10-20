# Data

## TT100K, 45-class protocol (2016 annotations)
- The 45 categories with more than 100 instances in the official training and test sets; all other categories are background.
- The official test set (3,071 images) is the test set. Six images that are byte-identical in the official train and test folders are kept in the test set and removed from training (`splits/excluded_train_overlap_ids.txt`).
- Validation set: 610 images (10% of the remaining training images, seed 0, every class represented). Training set: 5,489 images.
- `splits/{train,val,test}_ids.txt`: image IDs (file stems). `splits/calib_train500_seed0.txt`: the 500 training images used for INT8 calibration.
- Preparation: `scripts/prepare_tt100k.py`; download links in `DOWNLOAD.md`. Images are not included.

## CCTSDB 2021 (cross-dataset test)
- Official test set (1,500 images, 3,228 signs), prepared with `scripts/prepare_cctsdb_test.py`.
- `cctsdb/tt100k45_to_cctsdb3.csv`: mapping of the 45 TT100K classes to the three CCTSDB super-classes by prefix (p → prohibitory, w → warning, i → mandatory).
