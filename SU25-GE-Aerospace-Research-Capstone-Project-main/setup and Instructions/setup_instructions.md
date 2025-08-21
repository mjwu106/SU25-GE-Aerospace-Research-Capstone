- # Dataset Setup Instructions

  ## What You Need

  - Python 3.7+
  - 12GB free disk space
  - Good internet connection

  ## Steps

  ### 1. Clone Repository

  ```bash
  git clone https://github.com/mjwu106/SU25-GE-Aerospace-Research-Capstone
  cd SU25-GE-Aerospace-Research-Capstone-Project
  ```

  ### 2. Download Datasets

  - Go to Google Drive: [[Google Drive Link](https://drive.google.com/file/d/1JOa-sM46Zz_NVdQntAZ-bg3ZR4oZnLqu/view?usp=drive_link)]
  - Download the dataset files
  - Extract to `datasets/data/` folder

  ### 3. Verify Setup

  ```bash
  python datasets/verify.py
  ```

  ## Expected Folder Structure

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

  ## Original Sources

  - VisDrone: https://github.com/VisDrone/VisDrone-Dataset
  - BDD100K: https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k