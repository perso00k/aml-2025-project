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
│   ├── checkpoints/   -> Official EgoVLP repository: https://drive.google.com/file/d/1-cP3Gcg0NGDcMZalgJ_615BQdbFIbcj7/view 
│   │   └── egovlp.pth
│   ├── EgoVLP-main/   -> Official EgoVLP repository: https://github.com/showlab/EgoVLP 
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
│   ├── features/   -> Output directory. If you want you can download the extracted features from our Google Drive.  
│   └── videos/    
│       └── ... (raw video files) -> Download the videos here: https://drive.google.com/file/d/18rWqGNUzXqlJjLvYVcbeHTJsd4lsJ0Rb/view
├── annotations-main/  -> You can clone files from the official repository: https://github.com/CaptainCook4D/annotations/tree/main 
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
│   │   ├── HiERO/  -> You can clone files from the official repository: https://github.com/sapeirone/HiERO 
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
│   │   ├── steps/ -> Output directory. If you want you can download it from our Google Drive. 
│   │   └── video_params_dump.csv  -> You can find this file into this repository in    colab_notebooks/Extension_Part/Substep_1
│   ├── step_2_baseline/
│   │   └── model_result/ -> Output directory. 
│   ├── step_3_task_graph/ 
│   │   ├── gnn_ready_data/ -> Output directory. If you want you can download it from our Google Drive.
│   │   ├── matched_features/ -> Output directory. If you want you can download it from our Google Drive.
│   │   └── text_features_egovlp/ Output directory. If you want you can download it from our Google Drive. 
│   └── step_4_gnn/
│       └── gnn_ready_data_groundtruth/ Output directory. If you want you can download it from our Google Drive.
└── First_Part/
    ├── features/ -> You can download the zips from here: https://utdallas.app.box.com/s/zzuglo0j0loo8ymdsxfr2zzgf9q2jajc/folder/291562599734 
    │   ├── omnivore.zip 
    │   └── slowfast.zip
    ├── models_result_omnivore/ -> Output directory.
    ├── models_result_slowfast/ -> Output directory.
    └── error_recognition_best.zip -> You can download it from https://utdallas.app.box.com/s/uz3s1alrzucz03sleify8kazhuc1ksl3/folder/299366346861 
```
## How to Run the Code

The project is divided into **modular steps**. Please execute the notebooks in the following logical order if you start from scratch:

### 1. Baselines (Mistake Detection)

**Notebook:** `colab_notebooks\First_Part\Omnivore.ipynb` 

**Notebook:** `colab_notebooks\First_Part\Slowfast.ipynb` 

**Description:**
These notebooks independently execute the supervised learning pipeline for step and recordings level mistake detection. They share the same logic but utilize different pre-extracted feature backbones to train **MLP**, **Transformer**, and **LSTM** classifiers.

- **Input:** They automatically load the corresponding feature sets from `First_Part/features/` (e.g., `omnivore.zip` or `slowfast.zip`)
- **Output:** Performance metrics, confusion matrices, and CSV reports are saved to `First_Part/models_result_omnivore/` and `First_Part/models_result_slowfast/` respectively

### 2. Feature Extraction (Extension)

**Notebook:** `colab_notebooks\First_Part\EgoVLP_video_features.ipynb`

**Description:**
This notebook manages the extraction of high-level semantic features from raw video data using the **EgoVLP** backbone.

**Input:** 
- Raw Videos (`3_EgoVLP/videos`) 
- EgoVLP Repository (`3_EgoVLP/EgoVLP-main`)
- EgoVLP Checkpoint (`3_EgoVLP/checkpoints/egovlp.pth`)

**Output:** 
- Extrected Features (`3_EgoVLP/features/`)

### 3. Substep 1 - Step Localization (Extension)

**Notebook:** `colab_notebooks\Extension_Part\Substep_1\/HiERO.ipynb`

**Description:**
This notebook performs unsupervised temporal segmentation of the video features to identify recipe steps.

**Input:**
- The features extracted in Step 2 (from `3_EgoVLP/features/`)
- **Crucial:** The `video_params_dump.csv` file, which contains the target number of clusters (`n_clusters`) and FPS metadata for each video. 

**Output:** It saves the segmentation results (segments and averaged step embeddings) as `.npz` files in `Extension/step_1_HiERO/steps/`

### 4. Substep 2 - Simple Task Verification Baseline (Extension)

**Notebook:** `colab_notebooks\Extension_Part\Substep_2\baseline.ipynb`

**Description:**
This notebook trains a baseline to perform the Task Verification task without explicitly using the Task Graph structure.

**Input:** The sequence of step-level embeddings from `Extension/step_1_HiERO/steps/`

**Output:** Model checkpoints and evaluation plots (confusion matrices, accuracy) saved in `Extension/step_2_baseline/model_result/`

### 5. Substep 3 - Task Graph Alignment (Extension)

**Notebook:** `colab_notebooks\Extension_Part\Substep_3\Substep3.ipynb`

**Description:**
This notebook matches videos and text features and binds them to the respective recipe graph.

**Input:**
- Video step embeddings (`Extension/step_1_HiERO/steps/`)
- Task Graph JSON files (`annotations-main/task_graphs/`)
- Error annotation JSON file (`annotations-main/annotation_json/error_annotations.json`)

**Output:** It saves the graphs into `Extension/step_3_task_graph/gnn_ready_data`

### 6. Substep 4 - Task Verification via GNNs (Extension)

**Notebook:** `colab_notebooks\Extension_Part\Substep_4\Substep4V1.ipynb`

**Description:**
This is the final classification stage. We use **Graph Neural Networks (GNNs)** to process the aligned task graphs and predict if the recipe execution was correct. This step is split into three notebooks to separate data generation, oracle evaluation, and actual pipeline evaluation.

**Input:**
- Graphs (`Extension/step_3_task_graph/gnn_ready_data`)

### 6.1 Substep 4.b - Task Verification via GNNs on Grount Truth Graphs (Extension)

**Notebook Graph Creation:** `colab_notebooks\Extension_Part\Substep_4\GroundTruth_GraphCreation.ipynb`
**Notebook Classification:** `colab_notebooks\Extension_Part\Substep_4\Substep4_onGT.ipynb`

**Description:**
Here we reconstruct the graphs using the annotations in the Ground Truth and we use them to train the GNN. In this way we can compare the results with the ones obtained by our model. 

**Input Graph Creation:**
- Task Graph JSON files (`annotations-main/task_graphs/`)
- Error annotation JSON file (`annotations-main/annotation_json/error_annotations.json`)
- Error annotation JSON file (`annotations-main/annotation_json/complete_step_annotations.json`)
- Features Video (`3_EgoVLP/features`)
- Features Text (`Extension/step_3_task_graph/text_features_egovlp`)

**Output Graph Creation / Input Classification**
- Graphs (`Extension/step_4_gnn/gnn_ready_data_groundtruth`)

---