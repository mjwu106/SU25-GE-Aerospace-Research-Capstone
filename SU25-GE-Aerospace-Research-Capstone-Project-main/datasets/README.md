- # Project Datasets

  ## Overview

  - **VisDrone**: UAV object detection dataset (~1.8GB)
  - **BDD100K**: UGV driving scene dataset (~7GB)

  ## Quick Setup

  1. Download datasets from [[Google Drive Link](https://drive.google.com/file/d/1JOa-sM46Zz_NVdQntAZ-bg3ZR4oZnLqu/view?usp=drive_link)]
  2. Extract to `datasets/data/` directory
  3. Run: `python datasets/verify.py`

  ## Dataset Info

  ### VisDrone

  - Training: 6,471 images
  - Validation: 548 images
  - Test: 1,610 images
  - Format: txt files

  ### BDD100K

  - Training: ~70,000 images
  - Validation: ~10,000 images
  - Format: JSON files

  ## Directory Structure

  ```
  datasets/
  ├── data/
  │   ├── BDD100K/
  │   │   ├── bdd100k/bdd100k/images/100k/
  │   │   │   ├── train/
  │   │   │   ├── val/
  │   │   │   └── test/
  │   │   └── bdd100k_labels_release/bdd100k/labels/
  │   └── visDrone/
  │       ├── VisDrone2019-DET-train/
  │       │   ├── annotations/
  │       │   └── images/
  │       ├── VisDrone2019-DET-val/
  │       │   ├── annotations/
  │       │   └── images/
  │       └── VisDrone2019-DET-test-dev/
  │           ├── annotations/
  │           └── images/
  ├── README.md
  ├── download.py
  ├── verify.py
  └── setup_instructions.md
  ```

  ## Sources

  - VisDrone: https://github.com/VisDrone/VisDrone-Dataset
  - BDD100K: https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k