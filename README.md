# Procedural Mistake Detection & Task Verification

## Project Overview

This project focuses on **Procedure Understanding** using egocentric videos. Specifically, we address two main tasks:

- **Mistake Detection**: Identifying errors in individual steps of a recipe (using baselines like SlowFast and Omnivore)
- **Task Verification (Extension)**: Determining if an entire recipe execution is correct by aligning video steps with a Task Graph

The implementation includes feature extraction using **EgoVLP**, step localization via **HiERO**, and a final graph-based verification using **GNNs**.

## Setup & Data Management

This project was developed using **Google Colab** to leverage GPU resources. Consequently, the data pipeline is designed to integrate directly with **Google Drive**.

To run the notebooks successfully, you must ensure the directory structure in your Google Drive matches the paths defined in the code. You have two options to set this up:

### Option A: Direct Drive Access (Recommended)

We have hosted the entire project structure (including datasets, checkpoints, and features) on a shared Google Drive folder.

**Access the Shared Folder:**
- Click the following link to access the project resources: **[Link to Shared Google Drive Folder](https://drive.google.com/drive/folders/1vxgD6uYnr2LpQalDA9eOLzCE7O5v_LHt?usp=sharing)**

**Create a Shortcut (Crucial Step):**
1. Go to the **"Shared with me"** section in your Google Drive
2. Right-click on the folder named **`AML_Project`**
3. Select **Organize > Add shortcut**
4. Navigate to **My Drive** and place the shortcut directly in the root

**Important**: The notebooks mount Drive at `/content/drive/MyDrive/`. If the `AML_Project` folder is not directly reachable via this path, the code will fail to load files.

### Option B: Manual Setup

If you cannot use the shared link, you must manually recreate the exact directory structure inside the root of your Google Drive (`MyDrive`). Ensure your folder structure looks exactly like the tree below. Any deviation in folder naming or nesting will cause `FileNotFoundError` in the notebooks.

```text
AML_Project/
├── 3_EgoVLP/
│   ├── checkpoints/
│   │   └── egovlp.pth
│   ├── EgoVLP-main/
│   │   ├── base/
│   │   ├── configs/
│   │   ├── data_loader/
│   │   ├── figures/
│   │   ├── logger/
│   │   ├── model/
│   │   ├── run/
│   │   ├── trainer/
│   │   ├── utils/
│   │   ├── environment.yml
│   │   ├── parse_config.py
│   │   └── README.md
│   ├── features/
│   │   ├── 10_16_360p_224.mp4_1s_1s.npy
│   │   ├── 10_16_360p_224.mp4_1s_1s.npz
│   │   └── ... (extracted features)
│   ├── pretrained/
│   │   ├── distilbert-base-uncased/
│   │   └── jx_vit_base_p16_224-80ecf9dd.pth
│   ├── videos/
│   │   └── ... (raw video files)
│   └── EgoVLP_video_features.ipynb
├── annotations-main/
│   ├── annotation_csv/
│   ├── annotation_json/
│   ├── data_splits/
│   ├── metadata/
│   ├── task_graphs/
│   ├── ANNOTATIONS.md
│   ├── LICENSE
│   └── README.md
├── Extension/
│   ├── step_1_HiERO/
│   │   ├── HiERO/
│   │   │   ├── assets/
│   │   │   ├── checkpoints/
│   │   │   ├── configs/
│   │   │   ├── data/
│   │   │   ├── ego4d_goalstep/
│   │   │   ├── egoprocel/
│   │   │   ├── features-extraction/
│   │   │   ├── models/
│   │   │   ├── utils/
│   │   │   ├── LICENSE
│   │   │   ├── quickstart.ipynb
│   │   │   └── README.md
│   │   ├── steps/
│   │   ├── HiERO.ipynb
│   │   └── video_params_dump.csv
│   ├── step_2_baseline/
│   │   ├── model_result/
│   │   └── baseline.ipynb
│   ├── step_3_task_graph/
│   │   ├── gnn_ready_data/
│   │   ├── matched_features/
│   │   ├── pretrained/
│   │   ├── text_features_egovlp/
│   │   └── Substep3.ipynb
│   └── step_4_gnn/
│       ├── gnn_ready_data_groundtruth/
│       ├── GroundTruth_GraphCreation.ipynb
│       ├── Substep4_onGT.ipynb
│       └── Substep4V1.ipynb
├── First_Part/
│   ├── features/
│   │   ├── omnivore.zip
│   │   └── slowfast.zip
│   ├── models_result_omnivore/
│   ├── models_result_slowfast/
│   ├── error_recognition_best.zip
│   ├── Omnivore.ipynb
│   └── Slowfast.ipynb
└── AML-2025_Mistake_Detection_Project.gdoc
```
## How to Run the Code

The project is divided into **modular steps**. Please execute the notebooks in the following logical order:

### 1. Baselines (Mistake Detection)

**Location:** `First_Part/` directory  
**Files:** `Omnivore.ipynb` and `Slowfast.ipynb`

**Description:**
These notebooks independently execute the supervised learning pipeline for step-level mistake detection. They share the same logic but utilize different pre-extracted feature backbones to train **MLP**, **Transformer**, and **LSTM** classifiers.

- **Input:** They automatically load the corresponding feature sets from `First_Part/features/` (e.g., `omnivore.zip` or `slowfast.zip`)
- **Output:** Performance metrics, confusion matrices, and CSV reports are saved to `First_Part/models_result_omnivore/` and `First_Part/models_result_slowfast/` respectively

### 2. Feature Extraction (Extension)

**Location:** `3_EgoVLP/` directory  
**File:** `EgoVLP_video_features.ipynb`

**Description:**
This notebook manages the extraction of high-level semantic features from raw video data using the **EgoVLP** backbone.

- **Process:** It first checks for the `EgoVLP-main` repository and installs necessary dependencies. It then processes raw videos located in `3_EgoVLP/videos/`
- **Output:** It generates compressed feature files (`.npz`) sampled at 1-second intervals, saving them to `3_EgoVLP/features/`. These features are the prerequisites for the step localization in Step 3

### 3. Substep 1 - Step Localization (Extension)

**Location:** `Extension/step_1_HiERO/` directory  
**Files:** `HiERO.ipynb` and `video_params_dump.csv`

**Description:**
This notebook performs unsupervised temporal segmentation of the video features to identify recipe steps.

**Input:**
- The features extracted in Step 2 (from `3_EgoVLP/features/`)
- **Crucial:** The `video_params_dump.csv` file, which contains the target number of clusters (`n_clusters`) and FPS metadata for each video

**Process:** It loads the pre-trained HiERO model to compute hierarchical features, then applies spectral clustering to determine step boundaries (start_time, end_time)

**Output:** It saves the segmentation results (segments and averaged step embeddings) as `.npz` files in `Extension/step_1_HiERO/steps/`

### 4. Substep 2 - Simple Task Verification Baseline (Extension)

**Location:** `Extension/step_2_baseline/` directory  
**File:** `baseline.ipynb`

**Description:**
This notebook trains a baseline model (e.g., a Transformer or MLP classifier) to perform the Task Verification task without explicitly using the Task Graph structure.

**Concept:** It tries to predict if a recipe execution is correct solely by looking at the sequence of visual step embeddings generated in Step 3. Think of this as trying to guess if a sentence is grammatically correct by only looking at the list of words, without applying explicit grammar rules.

- **Input:** The sequence of step-level embeddings from `Extension/step_1_HiERO/steps/`
- **Output:** Model checkpoints and evaluation plots (confusion matrices, accuracy) saved in `Extension/step_2_baseline/model_result/`

### 5. Substep 3 - Task Graph Alignment (Extension)

**Location:** `Extension/step_3_task_graph/` directory  
**File:** `Substep3.ipynb`

**Description:**
This notebook aligns the localized video steps (visual domain) with the textual Task Graph nodes (text domain) provided in the annotations.

**Concept:** It acts as a bridge between vision and text. It uses the **Hungarian Matching** algorithm to find the optimal one-to-one pairing between the detected video segments and the recipe instructions based on feature similarity.

**Input:**
- Video step embeddings (`Extension/step_1_HiERO/steps/`)
- Task Graph JSON files (`annotations-main/task_graphs/`)

**Output:** It saves the aligned features (where each video step is "tagged" with its corresponding task node) to `Extension/step_3_task_graph/matched_features/`

### 6. Substep 4 - Task Verification via GNNs (Extension)

**Location:** `Extension/step_4_gnn/` directory

**Description:**
This is the final classification stage. We use **Graph Neural Networks (GNNs)** to process the aligned task graphs and predict if the recipe execution was correct. This step is split into three notebooks to separate data generation, oracle evaluation, and actual pipeline evaluation.

---