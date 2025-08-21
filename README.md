# Comprehensive Analysis of Data Augmentation Techniques Impacting YOLO and DETR Object Detectors

## Purpose

The goal of this project is to systematically evaluate and compare a broad array of image augmentation techniques to quantify their impact on YOLO and DETR object detector performance, with the aim of identifying the most effective approaches for achieving robust, reliable detection in autonomous systems.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Preparation](#dataset-preparation)
- [Environment Setup](#environment-setup)
- [Configuration Files](#configuration-files)
- [Task Implementations](#task-implementations)
  - [Task 2 & 6: Out-of-the-box Validation](#task-2--6-out-of-the-box-validation)
  - [Task 3-5: Fine-tuning Experiments](#task-3-5-fine-tuning-experiments)
  - [Task 7: PR Curve Analysis](#task-7-pr-curve-analysis)
  - [Task 8: mAP Performance Table](#task-8-map-performance-table)
  - [Task 9: Combined Dataset Training](#task-9-combined-dataset-training)
- [Results Analysis](#results-analysis)
- [Hardware Requirements](#hardware-requirements)

## Project Overview

This project conducts a comprehensive evaluation of data augmentation techniques on object detection models, specifically:

- **Models**: YOLOv11 and RT-DETR
- **Datasets**: BDD100K and VisDrone
- **Augmentation Techniques**: Horizontal flip, vertical flip, and intensity channel conversion
- **Evaluation Metrics**: mAP@0.5, mAP@0.5:0.95, Precision-Recall curves

## Dataset Preparation

### Original Dataset Conversion

The VisDrone dataset has been converted from its original format to YOLO format for compatibility with the ultralytics framework. The conversion handles:

- Category mapping (VisDrone categories 1-10 → YOLO categories 0-9)
- Coordinate normalization
- Invalid region filtering (score=0 regions are skipped)

```python
import os
from PIL import Image

subsets = [
    "VisDrone2019-DET-test-dev",
    "VisDrone2019-DET-train",
    "VisDrone2019-DET-val",
]

base_path = "datasets/data/visDrone"

for subset in subsets:
    anno_dir = os.path.join(base_path, subset, "annotations")
    label_dir = os.path.join(base_path, subset, "labels")
    img_dir = os.path.join(base_path, subset, "images")
    os.makedirs(label_dir, exist_ok=True)

    for file in os.listdir(anno_dir):
        if not file.endswith(".txt"):
            continue

        anno_path = os.path.join(anno_dir, file)
        label_path = os.path.join(label_dir, file)

        # Get the actual dimensions of the corresponding image
        img_name = file.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)

        try:
            img = Image.open(img_path)
            img_width, img_height = img.size
        except:
            print(f"Cannot open image: {img_path}")
            continue

        with open(anno_path, "r") as fin, open(label_path, "w") as fout:
            for line in fin:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue

                x, y, w, h, score, cls_id, truncation, occlusion = map(int, parts[:8])

                # Skip ignored regions (score=0)
                if score == 0:
                    continue

                # Skip invalid categories (VisDrone categories are 1-10, converted to 0-9)
                if cls_id < 1 or cls_id > 10:
                    continue

                # Category ID minus 1 (VisDrone: 1-10 -> YOLO: 0-9)
                yolo_cls_id = cls_id - 1

                # Normalize coordinates
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height

                # Ensure coordinates are within a valid range
                if (
                    0 <= x_center <= 1
                    and 0 <= y_center <= 1
                    and w_norm > 0
                    and h_norm > 0
                ):
                    fout.write(
                        f"{yolo_cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                    )

print("Conversion complete!")
```

### Pre-processed Dataset Access

You can directly access the converted datasets (original + augmented with horizontal flip, vertical flip, and intensity channel conversion) via Google Drive:

**Download Link**: [https://drive.google.com/file/d/1XWLFi_AKVRgC5oTwlzvZBUgCppuR0caH/view](https://drive.google.com/file/d/1XWLFi_AKVRgC5oTwlzvZBUgCppuR0caH/view)

To quickly download the dataset to your server:

```bash
pip install gdown
gdown https://drive.google.com/file/d/1XWLFi_AKVRgC5oTwlzvZBUgCppuR0caH/view
# Unzipping will take approximately 1 hour
```

## Environment Setup

### Hardware Requirements

- **Recommended**: ICRN PyTorch server with A100 GPU access
- **GPU Monitoring**: Use `watch -n 0.5 nvidia-smi` to monitor GPU status
- **Platform**: Jupyter notebook (`.ipynb`) is recommended for interactive development

### Required Packages

```python
!pip install torch torchvision torchaudio opencv-python-headless ultralytics numpy==1.26.4
```

## Configuration Files

Both tasks require YAML configuration files for dataset specification.

### BDD100K Configuration (`bdd100k.yaml`)

```yaml
names:
  0: bike
  1: bus
  2: car
  3: drivable area
  4: lane
  5: motor
  6: person
  7: rider
  8: traffic light
  9: traffic sign
  10: train
  11: truck
nc: 12
path: ../../datasets/data/BDD100K/yolo
train: train/images
val: val/images
```

### VisDrone Configuration (`visdrone.yaml`)

```yaml
names:
- pedestrian
- people
- bicycle
- car
- van
- truck
- tricycle
- awning-tricycle
- bus
- motor
nc: 10
path: ../../datasets/data/visDrone
train: VisDrone2019-DET-train/images
val: VisDrone2019-DET-val/images
```

### Multi-Dataset Configuration (Task 9)

For combined dataset training, modify the train path:

```yaml
train: 
  - augmented_data/BDD100K/horizontal_flip/train/images
  - augmented_data/BDD100K/vertical_flip/train/images
  - augmented_data/BDD100K/intensity_channel/train/images
```

## Task Implementations

### Task 2 & 6: Out-of-the-box Validation

Evaluate pre-trained models on original and augmented datasets without fine-tuning.

```python
# Import libraries
from ultralytics import YOLO, RTDETR
from datetime import datetime
import pandas as pd
import os

# Configuration for all experiments
experiments = [
    {"model": "YOLO", "model_file": "yolo11n.pt", "data": "bdd100k.yaml", "name": "YOLO_BDD100k"},
    {"model": "YOLO", "model_file": "yolo11n.pt", "data": "visdrone.yaml", "name": "YOLO_VisDrone"},
    {"model": "RTDETR", "model_file": "rtdetr-l.pt", "data": "bdd100k.yaml", "name": "RTDETR_BDD100k"},
    {"model": "RTDETR", "model_file": "rtdetr-l.pt", "data": "visdrone.yaml", "name": "RTDETR_VisDrone"}
]

if __name__ == "__main__":
    for exp in experiments:
        print(f"\nEvaluating {exp['name']}...")
        
        # Load a model
        if exp["model"] == "YOLO":
            model = YOLO(exp["model_file"])
        else:
            model = RTDETR(exp["model_file"])
        
        # Validate the model
        results = model.val(
            data=exp["data"],
            imgsz=640,
            batch=16,
            device=0,
            workers=0,
            cache=False,
        )
        
        # You can print or process results here
        print("Validation Results:")
        print(results)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"validation_results_{exp['name']}_{timestamp}.txt"
        with open(results_file, "w") as f:
            f.write(f"Validation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Validation Results: {str(results)}\n")
        
        print(f"Results saved to: {results_file}")

        # Save PR curve to CSV
        try:
            pr_data = results.box.pr_curve  # shape (num_classes, 100, 2)
            os.makedirs("pr_curves", exist_ok=True)

            for class_idx, class_curve in enumerate(pr_data):
                df = pd.DataFrame(class_curve, columns=["Recall", "Precision"])
                csv_name = f"pr_curves/{exp['name']}_class{class_idx}_pr_curve.csv"
                df.to_csv(csv_name, index=False)
                print(f"Saved PR curve CSV for class {class_idx} to {csv_name}")

        except Exception as e:
            print(f"⚠️ Failed to save PR curve CSV: {e}")
```

**Expected Output**: 4 validation result files with mAP metrics for Task 8 analysis.

### Task 3-5: Fine-tuning Experiments

Train models on different dataset configurations:
- **Task 3**: Fine-tune on original data (20%) → Test on original
- **Task 4**: Fine-tune on augmented data (20%) → Test on original  
- **Task 5**: Fine-tune on augmented data (20%) → Test on augmented

```python
# Import libraries
from ultralytics import YOLO, RTDETR
import multiprocessing
from datetime import datetime

# Configuration for all experiments
experiments = [
    {"model": "YOLO", "model_file": "yolo11n.pt", "data": "bdd100k.yaml", "name": "YOLO_BDD100k"},
    {"model": "YOLO", "model_file": "yolo11n.pt", "data": "visdrone.yaml", "name": "YOLO_VisDrone"},
    {"model": "RTDETR", "model_file": "rtdetr-l.pt", "data": "bdd100k.yaml", "name": "RTDETR_BDD100k"},
    {"model": "RTDETR", "model_file": "rtdetr-l.pt", "data": "visdrone.yaml", "name": "RTDETR_VisDrone"}
]

if __name__ == "__main__":
    for exp in experiments:
        print(f"\nTraining {exp['name']}...")
        
        # Load a model
        if exp["model"] == "YOLO":
            model = YOLO(exp["model_file"])
        else:
            model = RTDETR(exp["model_file"])
        
        # Train the model
        results = model.train(
            data=exp["data"],
            fraction=0.2,
            epochs=100,
            patience=10,
            imgsz=640,
            batch=16,
            device=0,
            workers=0,
            cache=False,
            augment=False,
            save=True,
            project="fine_tuning_results",
        )
        
        print("\nEvaluating model performance...")
        val_results = model.val(
            data=exp["data"],
            imgsz=640,
            batch=16,
            device=0,
            workers=0,
            cache=False,
        )
        
        # You can print or process results here
        print("Training Results:")
        print(results)
        print("\nValidation Results:")
        print(val_results)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"training_results_{exp['name']}_{timestamp}.txt"
        with open(results_file, "w") as f:
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training Results: {str(results)}\n")
            f.write(f"Validation Results: {str(val_results)}\n")
        
        print(f"Results saved to: {results_file}")
```

**Training Duration**: Approximately 10 hours for all experiments. Ensure your server session remains active and configure power settings to prevent sleep/hibernation.

**Output**: 4 result files and a `fine_tuning_results` directory containing trained models and automatically generated performance plots.

### Task 7: PR Curve Analysis

#### Step 1: Modify ultralytics Package

To extract detailed PR curve data, modify the ultralytics metrics module:

1. Locate the ultralytics installation:
```bash
pip show ultralytics
```

2. Navigate to `/ultralytics/utils/metrics.py`

3. Find the `plot_pr_curve` function and insert the following code between `ax.set_title(...)` and `fig.savefig(...)`:

```python
# === Save PR curves to CSV ===
import os
csv_dir = save_dir.parent / "pr_csv"
os.makedirs(csv_dir, exist_ok=True)

if 0 < len(names) < 21:
    for i, y in enumerate(py.T):
        with open(csv_dir / f"{names[i]}_Box.csv", "w") as f:
            f.write("recall,precision\n")
            for r, p in zip(px, y):
                f.write(f"{r},{p}\n")

# Save average (all classes) curve
with open(csv_dir / "all_classes.csv", "w") as f:
    f.write("recall,precision\n")
    for r, p in zip(px, py.mean(1)):
        f.write(f"{r},{p}\n")
```

#### Step 2: Generate PR Curves for Fine-tuned Models

```python
from ultralytics import YOLO, RTDETR
from datetime import datetime

# Configure information for each model
experiments = [
    {"model": "YOLO", "weights": "fine_tuning_results/train/weights/best.pt", "data": "bdd100k.yaml", "name": "YOLO_BDD_aug"},
    {"model": "RTDETR", "weights": "fine_tuning_results/train3/weights/best.pt", "data": "bdd100k.yaml", "name": "RTDETR_BDD_aug"},
    {"model": "RTDETR", "weights": "fine_tuning_results/train4/weights/best.pt", "data": "visdrone.yaml", "name": "RTDETR_Vis_aug"},
    {"model": "YOLO", "weights": "fine_tuning_results/train6/weights/best.pt", "data": "visdrone.yaml", "name": "YOLO_Vis_aug"},
]

for exp in experiments:
    # Load the model
    model = YOLO(exp["weights"]) if exp["model"] == "YOLO" else RTDETR(exp["weights"])

    # Validate the model, generate PR curve, confusion matrix, prediction visualizations, etc.
    results = model.val(
        data=exp["data"],
        imgsz=660,  # Changed from 640 to 660 as per common practice for better performance with some models.
        batch=16,
        device=0,
        workers=0,
        cache=False,
        name=exp["name"]
    )

    # Save the results log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"val_log_{exp['name']}_{timestamp}.txt", "w") as f:
        f.write(str(results))

    print(f"[{exp['name']}] validation completed.")
```

#### Step 3: Plot Comparative PR Curves

After collecting all `all_classes.csv` files from `runs/detect/{model_name}_{dataset_name}/pr_csv/`, organize them and create comparative plots:

```python
import os
import pandas as pd
import matplotlib.pyplot as plt

# Change to the PR_curve directory (modify as needed)
os.chdir("Fine_Tuning/PR_curve")

# Define two groups of tasks
tasks_original = {
    "2": "Pretrained on Original",
    "3": "FT Unaug (20%) → Test Original",
    "4": "FT Aug (20%) → Test Original",
}
tasks_augmented = {
    "5": "FT Aug (20%) → Test Augmented",
    "6": "Pretrained on Augmented",
}

# Model and dataset combinations
models = ["YOLO", "RTDETR"]
datasets = ["BDD100k", "VisDrone"]

# Universal plotting function
def plot_pr_curve(task_group, title_suffix, output_suffix):
    for dataset in datasets:
        plt.figure(figsize=(12, 8))
        for task_id, task_desc in task_group.items():
            for model in models:
                file_path = f"{task_id}/all_classes_{model}_{dataset}.csv"
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    try:
                        plt.plot(
                            df["recall"],
                            df["precision"],
                            label=f"{model} - {task_desc}",
                        )
                    except KeyError as e:
                        print(f"⚠️ Error in file {file_path}: Missing column {e}")
                        print("Actual columns are:", df.columns.tolist())
                else:
                    print(f"⚠️ Warning: File not found: {file_path}")

        plt.title(f"PR Curve Comparison on {title_suffix} ({dataset})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        # Set legend location to upper right and adjust font size for a relatively smaller legend box
        plt.legend(loc="upper right", prop={"size": 12}) 
        plt.grid(True)
        plt.tight_layout()
        save_name = f"stacked_pr_{output_suffix}_{dataset}.png"
        plt.savefig(save_name, dpi=300)
        print(f"✅ Saved {save_name}")
        plt.close()

# Plot curves for the original validation set
plot_pr_curve(
    tasks_original, title_suffix="Original Validation Set", output_suffix="original"
)

# Plot curves for the augmented validation set
plot_pr_curve(
    tasks_augmented, title_suffix="Augmented Validation Set", output_suffix="augmented"
)
```

### Task 8: mAP Performance Table

Extract mAP@0.5:0.95 values from all result files to create a comprehensive performance comparison table. Look for the `metrics/mAP50-95(B)` values in each validation result file.

Example result format:
```
results_dict: {'metrics/precision(B)': 0.10561918617238329, 'metrics/recall(B)': 0.09774884853543368, 'metrics/mAP50(B)': 0.09823072288034473, 'metrics/mAP50-95(B)': 0.05707781506110854, 'fitness': 0.06119310584303216}
```

### Task 9: Combined Dataset Training

Train models on combined datasets using 10% of each dataset.

**Configuration**: Modify the YAML file to include multiple augmented dataset paths as shown in the configuration section.

**Training Code**: Use the fine-tuning code with `fraction=0.1` instead of `fraction=0.2`.

## Results Analysis

### Performance Metrics

The project evaluates models using:
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds 0.5 to 0.95
- **Precision-Recall Curves**: Detailed performance analysis across different confidence thresholds

### Expected Outcomes

1. **Quantitative Analysis**: Performance improvements from different augmentation techniques
2. **Model Comparison**: YOLOv11 vs RT-DETR performance characteristics  
3. **Dataset Impact**: BDD100K vs VisDrone training effectiveness
4. **Augmentation Effectiveness**: Impact of horizontal flip, vertical flip, and intensity channel conversion

## Hardware Requirements

- **GPU**: NVIDIA A100 (recommended) or equivalent high-performance GPU
- **Memory**: Sufficient RAM for batch processing (16+ GB recommended)
- **Storage**: Adequate space for datasets, models, and results (~100GB+)
- **Compute Time**: Budget ~10 hours for complete fine-tuning experiments

## Important Notes

⚠️ **Power Management**: Configure your system to prevent sleep/hibernation during long training sessions.

⚠️ **Session Management**: Use screen/tmux for long-running processes on remote servers.

⚠️ **Backup**: Regularly save intermediate results and trained models.

This comprehensive framework provides a systematic approach to evaluating the impact of data augmentation techniques on modern object detection architectures, contributing valuable insights for autonomous system development.

