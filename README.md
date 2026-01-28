# AML 2025 - Progetto di Rilevazione Errori nelle Ricette

Questo progetto implementa un sistema di **rilevazione di errori in video di ricette egocentriche** utilizzando una combinazione di **Graph Neural Networks (GNNs)**, feature video e testuali estratte da modelli pre-addestrati.

## 📋 Panoramica del Progetto

Il progetto affronta la seguente sfida: dato un video egocentrico di una ricetta, determinare se **l'esecutore ha commesso errori** durante la preparazione.

### Architettura Generale
1. **Feature Video** (EgoVLP + HiERO): Estrazione di embeddings video per ogni step della ricetta
2. **Feature Testuali** (EgoVLP): Estrazione di embeddings semantici dagli step testuali delle ricette
3. **Allineamento Multimodale** (Hungarian Algorithm + Temporal Cost): Matching tra video e testo
4. **Graph Neural Network (GraphSAGE)**: Classificazione binaria (errore/no errore) basata sul grafo della ricetta

---

## 📁 Struttura del Google Drive (IMPORTANTE)

Per replicare correttamente il progetto, il professore deve creare la seguente struttura su Google Drive:

### Link Condiviso (fittizio per ora)
```
https://drive.google.com/drive/folders/1A2B3C4D5E6F7G8H9I0J1K2L3M4N5O6P7?usp=sharing
```
*Nota: Sostituire con il link reale quando il progetto è pronto per la consegna*

### Struttura Gerarchica Completa

```
AML_Project/
├── 3_EgoVLP/
│   ├── checkpoints/
│   │   └── egovlp.pth
│   ├── EgoVLP-main/
│   │   ├── base/
│   │   │   ├── __init__.py
│   │   │   ├── base_data_loader.py
│   │   │   ├── base_dataset.py
│   │   │   └── ... (2 other files)
│   │   ├── configs/
│   │   │   ├── eval/
│   │   │   │   ├── charades.json
│   │   │   │   ├── egomcq.json
│   │   │   │   ├── epic.json
│   │   │   │   └── ... (2 other files)
│   │   │   ├── ft/
│   │   │   │   ├── charades.json
│   │   │   │   ├── epic.json
│   │   │   │   ├── oscc.json
│   │   │   │   └── ... (1 other files)
│   │   │   └── pt/
│   │   │       └── egoclip.json
│   │   ├── data_loader/
│   │   │   ├── __init__.py
│   │   │   ├── CharadesEgo_dataset.py
│   │   │   ├── ConceptualCaptions_dataset.py
│   │   │   └── ... (9 other files)
│   │   ├── figures/
│   │   │   ├── egomcq.jpg
│   │   │   └── egovlp_framework.jpg
│   │   ├── logger/
│   │   │   ├── __init__.py
│   │   │   ├── logger.py
│   │   │   ├── logger_config.json
│   │   │   └── ... (1 other files)
│   │   ├── model/
│   │   │   ├── __init__.py
│   │   │   ├── load_checkpoint.py
│   │   │   ├── loss.py
│   │   │   └── ... (3 other files)
│   │   ├── run/
│   │   │   ├── test_charades.py
│   │   │   ├── test_epic.py
│   │   │   ├── test_mq.py
│   │   │   └── ... (6 other files)
│   │   ├── trainer/
│   │   │   ├── __init__.py
│   │   │   ├── trainer_charades.py
│   │   │   ├── trainer_egoclip.py
│   │   │   └── ... (3 other files)
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── charades_meta.py
│   │   │   ├── custom_transforms.py
│   │   │   └── ... (9 other files)
│   │   ├── environment.yml
│   │   ├── parse_config.py
│   │   └── README.md
│   ├── features/
│   │   ├── 10_16_360p_224.mp4_1s_1s.npy
│   │   ├── 10_16_360p_224.mp4_1s_1s.npz
│   │   ├── 10_18_360p_224.mp4_1s_1s.npy
│   │   └── ... (765 other files)
│   ├── pretrained/
│   │   ├── distilbert-base-uncased/
│   │   │   └── models--distilbert-base-uncased/
│   │   │       ├── blobs/
│   │   │       ├── refs/
│   │   │       └── snapshots/
│   │   └── jx_vit_base_p16_224-80ecf9dd.pth
│   ├── videos/
│   │   ├── 10_16_360p_224.mp4
│   │   ├── 10_18_360p_224.mp4
│   │   ├── 10_24_360p_224.mp4
│   │   └── ... (381 other files)
│   └── EgoVLP_video_features.ipynb
├── annotations-main/
│   ├── annotation_csv/
│   │   ├── activity_idx_step_idx.csv
│   │   ├── activity_step_description.csv
│   │   ├── average_segment_length.csv
│   │   └── ... (5 other files)
│   ├── annotation_json/
│   │   ├── activity_idx_step_idx.json
│   │   ├── complete_step_annotations.json
│   │   ├── error_annotations (1).json
│   │   └── ... (7 other files)
│   ├── data_splits/
│   │   ├── environment_data_split_combined.json
│   │   ├── environment_data_split_normal.json
│   │   ├── person_data_split_combined.json
│   │   └── ... (6 other files)
│   ├── metadata/
│   │   ├── average_segment_length.csv
│   │   └── video_information.csv
│   ├── task_graphs/
│   │   ├── blenderbananapancakes.json
│   │   ├── breakfastburritos.json
│   │   ├── broccolistirfry.json
│   │   └── ... (21 other files)
│   ├── ANNOTATIONS.md
│   ├── LICENSE
│   └── README.md
├── Extension/
│   ├── step_1_HiERO/
│   │   ├── HiERO/
│   │   │   ├── assets/
│   │   │   │   ├── hiero.png
│   │   │   │   └── teaser_animated.gif
│   │   │   ├── checkpoints/
│   │   │   │   └── hiero_egovlp.pth
│   │   │   ├── configs/
│   │   │   │   ├── components/
│   │   │   │   ├── defaults.yaml
│   │   │   │   ├── egovlp.yaml
│   │   │   │   ├── lavila-l.yaml
│   │   │   │   └── ... (1 other files)
│   │   │   ├── data/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── egoclip.py
│   │   │   │   ├── egomcq.py
│   │   │   │   └── ... (1 other files)
│   │   │   ├── ego4d_goalstep/
│   │   │   │   ├── annotations/
│   │   │   │   ├── utils/
│   │   │   │   ├── eval_grounding.py
│   │   │   │   └── README.md
│   │   │   ├── egoprocel/
│   │   │   │   ├── baseline_eval.py
│   │   │   │   ├── configs.py
│   │   │   │   ├── evaluate.py
│   │   │   │   └── ... (3 other files)
│   │   │   ├── features-extraction/
│   │   │   │   ├── configs/
│   │   │   │   ├── models/
│   │   │   │   ├── scripts/
│   │   │   │   ├── extract.py
│   │   │   │   ├── pipe.py
│   │   │   │   └── README.md
│   │   │   ├── models/
│   │   │   │   ├── conv/
│   │   │   │   ├── ext/
│   │   │   │   ├── tasks/
│   │   │   │   ├── __init__.py
│   │   │   │   └── hiero.py
│   │   │   ├── utils/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── dataloading.py
│   │   │   │   ├── gradients.py
│   │   │   │   └── ... (3 other files)
│   │   │   ├── LICENSE
│   │   │   ├── quickstart.ipynb
│   │   │   ├── README.md
│   │   │   └── ... (3 other files)
│   │   ├── steps/
│   │   │   ├── 10_16_steps.npz
│   │   │   ├── 10_18_steps.npz
│   │   │   ├── 10_24_steps.npz
│   │   │   └── ... (381 other files)
│   │   ├── HiERO.ipynb
│   │   └── video_params_dump.csv
│   ├── step_2_baseline/
│   │   ├── model_result/
│   │   │   ├── best_model.pth
│   │   │   ├── dataset_split_verification.png
│   │   │   ├── master_split_ids.json
│   │   │   └── ... (1 other files)
│   │   └── baseline.ipynb
│   ├── step_3_task_graph/
│   │   ├── gnn_ready_data/
│   │   │   ├── gnn_ready_10_16.pt
│   │   │   ├── gnn_ready_10_18.pt
│   │   │   ├── gnn_ready_10_24.pt
│   │   │   └── ... (381 other files)
│   │   ├── matched_features/
│   │   │   ├── match_10_16.pt
│   │   │   ├── match_10_18.pt
│   │   │   ├── match_10_24.pt
│   │   │   └── ... (381 other files)
│   │   ├── pretrained/
│   │   │   └── jx_vit_base_p16_224-80ecf9dd.pth
│   │   ├── text_features_egovlp/
│   │   │   ├── blenderbananapancakes.pt
│   │   │   ├── breakfastburritos.pt
│   │   │   ├── broccolistirfry.pt
│   │   │   └── ... (21 other files)
│   │   └── Substep3.ipynb
│   └── step_4_gnn/
│       ├── gnn_ready_data_groundtruth/
│       │   ├── gnn_ready_gt_10_16.pt
│       │   ├── gnn_ready_gt_10_18.pt
│       │   ├── gnn_ready_gt_10_24.pt
│       │   └── ... (381 other files)
│       ├── GroundTruth_GraphCreation.ipynb
│       ├── Substep4_onGT.ipynb
│       └── Substep4V1.ipynb
├── First_Part/
│   ├── features/
│   │   ├── omnivore.zip
│   │   └── slowfast.zip
│   ├── models_result_omnivore/
│   │   ├── lstm_recordings/
│   │   │   ├── accuracy_plot.png
│   │   │   ├── confusion_matrix.png
│   │   │   ├── final_lstm_report.csv
│   │   │   └── ... (2 other files)
│   │   └── lstm_step/
│   │       ├── accuracy_plot.png
│   │       ├── confusion_matrix.png
│   │       ├── final_lstm_report.csv
│   │       └── ... (2 other files)
│   ├── models_result_slowfast/
│   │   ├── lstm_recordings/
│   │   │   ├── accuracy_plot.png
│   │   │   ├── confusion_matrix.png
│   │   │   ├── final_lstm_report.csv
│   │   │   └── ... (2 other files)
│   │   └── lstm_step/
│   │       ├── accuracy_plot.png
│   │       ├── confusion_matrix.png
│   │       ├── final_lstm_report.csv
│   │       └── ... (2 other files)
│   ├── error_recognition_best.zip
│   ├── Omnivore.ipynb
│   └── Slowfast.ipynb
└── AML-2025_Mistake_Detection_Project.gdoc

```

---

## 🔍 Descrizione Dettagliata dei Substep

### Substep 1: Estrazione Feature Video con HiERO e EgoVLP

**File Notebook**: `First_Part/EgoVLP_video_features.ipynb`, `First_Part/Omnivore.ipynb`, `First_Part/Slowfast.ipynb`, `Extension_Part/Substep_1/HiERO.ipynb`

**Relazione alla Consegna (AML-2025.pdf)**:
- Implementa l'**extraction di feature video** dalle ricette egocentriche
- Utilizza modelli pre-addestrati per estrarre embeddings semantici
- Applica **HiERO zero-shot** per segmentare automaticamente i video in step

**Cosa Viene Fatto**:
1. **Estrazione Feature EgoVLP**: Carica il modello pre-addestrato EgoVLP e estrae embeddings video di dimensione 256 con risoluzione temporale 1 frame/secondo
2. **Segmentazione HiERO**: Applica il modello HiERO zero-shot per identificare automaticamente i confini tra step nella ricetta
3. **Generazione Timestamp**: Estrae il timing di inizio e fine per ogni step rilevato
4. **Output**: File `.npz` contenenti embeddings e timestamp

**Output Generato**:
- `AML_Project/3_EgoVLP/features/{recording_id}_360p_224.mp4_1s_1s.npy` - Feature video raw
- `AML_Project/Extension/step_1_HiERO/steps/{recording_id}_steps.npz` - Step segmentati con embeddings

---

### Substep 2: Baseline Transformer e Split Consistente

**File Notebook**: `Extension_Part/Substep_2/baseline.ipynb`

**Relazione alla Consegna (AML-2025.pdf)**:
- Stabilisce una **baseline di performance** usando un modello Transformer semplice
- Fissa lo split train/val/test che verrà mantenuto **identico** in tutti gli step successivi
- Consente il confronto tra la baseline e il modello GNN proposto

**Cosa Viene Fatto**:
1. **Caricamento Feature**: Legge i file `.npz` prodotti dal Substep 1
2. **Definizione Split**: Crea lo split train/val/test e lo salva in `master_split_ids.json`
3. **Modello Transformer**: Implementa un modello Transformer per classificazione binaria
4. **Addestramento e Valutazione**: Addestra il modello e riporta metriche di baseline
5. **Salvataggio Split**: Genera il file critico `master_split_ids.json` usato dagli step successivi

**Output Generato**:
- `AML_Project/Extension/step_2_baseline/model_result/master_split_ids.json`
- Metriche baseline (accuracy, F1, precision, recall)

---

### Substep 3: Feature Testuali e Costruzione Grafi Multimodali

**File Notebook**: `Extension_Part/Substep_3/Substep3.ipynb`

**Relazione alla Consegna (AML-2025.pdf)**:
- Implementa l'**integrazione multimodale**: combinazione di feature video e testuali
- Costruisce i **grafi canonici** delle ricette usando i dati di annotazione
- Applica **allineamento ottimo** tra i segment video predetti e gli step testuali della ricetta
- Prepara i dati per la **Graph Neural Network**

**Cosa Viene Fatto**:
1. **Estrazione Feature Testuali**:
   - Carica il modello EgoVLP
   - Estrae embeddings (dim: 256) per ogni step testuale della ricetta
   - Salva in `step_3_task_graph/text_features_egovlp/`

2. **Allineamento Video-Testo** (Hungarian Algorithm):
   - Calcola matrice di similarità coseno tra feature video e testuali
   - Aggiunge **penalità temporale** per favorire l'ordine cronologico corretto
   - Risolve il problema di assegnamento ottimo con algoritmo Hungarian
   - Output: matching tra video segments e recipe steps

3. **Costruzione Grafi**:
   - Legge il grafo canonico della ricetta da `task_graphs/{recipe_id}.json`
   - Allinea le feature video agli step del grafo tramite matching
   - Esegue late fusion: concatena feature video e testuali
   - Salva grafi in formato PyTorch Geometric

4. **Output**: Grafi pronti per la GNN in `step_3_task_graph/gnn_ready_data/`

**Output Generato**:
- `AML_Project/Extension/step_3_task_graph/text_features_egovlp/{recipe_id}.pt`
- `AML_Project/Extension/step_3_task_graph/matched_features/match_{recording_id}.pt`
- `AML_Project/Extension/step_3_task_graph/gnn_ready_data/gnn_ready_{recording_id}.pt`

---

### Substep 4: Classificazione GNN e Analisi Comparativa

**File Notebook**: `Extension_Part/Substep_4/Substep4V1.ipynb`, `Substep4_onGT.ipynb`, `GroundTruth_GraphCreation.ipynb`

**Relazione alla Consegna (AML-2025.pdf)**:
- Implementa la **Graph Neural Network (GraphSAGE)** per classificazione di errori
- Confronta le performance su **grafi predetti** vs **grafi Ground Truth ideali**
- Analizza l'impatto della qualità del grafo sulle performance finali

**Cosa Viene Fatto**:

#### 4.1 Substep4V1.ipynb (Classificazione su Grafi Predetti)
- Carica i grafi costruiti dal Substep 3 (`gnn_ready_data/`)
- Implementa architettura **GraphSAGE** con:
  - 2 layer convoluzionali
  - Batch normalization
  - Hybrid pooling (concatenazione Max + Mean)
  - Classificazione binaria con BCEWithLogitsLoss
- Addestra il modello con early stopping
- Valuta su test set e genera confusion matrix

#### 4.2 GroundTruth_GraphCreation.ipynb (Preparazione Grafi Ideali)
- Legge le annotazioni Ground Truth da `error_annotations.json`
- Estrae il timing corretto di ogni step da `complete_step_annotations.json`
- Estrae feature video usando il timing GT (non predetto)
- Costruisce grafi "ideali" in `step_4_gnn/gnn_ready_data_groundtruth/`
- Questi grafi hanno feature perfettamente allineate agli step reali

#### 4.3 Substep4_onGT.ipynb (Valutazione su Grafi Ground Truth)
- Carica lo stesso modello GNN addestrato su grafi predetti
- Valuta le performance sugli grafi Ground Truth
- Confronto critico:
  - **Se performance GT >> performance predetti**: Il problema è nella qualità del grafo (Step 3)
  - **Se performance GT ≈ performance predetti**: Il modello GNN è maturo e robusto

**Output Generato**:
- `AML_Project/Extension/step_4_gnn/gnn_ready_data_groundtruth/gnn_ready_gt_{recording_id}.pt`
- `best_gnn_model.pth` - Modello GNN addestrato
- Metriche finali: Accuracy, F1-Score, Recall, Specificity, Confusion Matrix

---

## 🚀 Come Replicare il Progetto

### Prerequisiti
- Google Drive con almeno 50GB di spazio
- Google Colab con accesso a GPU
- Repository EgoVLP (link nel progetto)

### Step-by-Step

1. **Creare la struttura Google Drive** seguendo l'albero di cartelle descritto sopra

2. **Caricare i dati**:
   - Annotazioni JSON in `annotations-main/annotation_json/`
   - Grafi delle ricette in `annotations-main/task_graphs/`
   - Checkpoint EgoVLP in `3_EgoVLP/checkpoints/`

3. **Eseguire i Notebook in sequenza**:
   ```
   Substep 1 (HiERO.ipynb)
        ↓
   Substep 2 (baseline.ipynb) → genera master_split_ids.json
        ↓
   Substep 3 (Substep3.ipynb) → genera grafi predetti
        ↓
   Substep 4 (Substep4V1.ipynb) → addestra GNN
   
   + Parallelo:
   Substep 4 (GroundTruth_GraphCreation.ipynb) → grafi ideali
        ↓
   Substep 4 (Substep4_onGT.ipynb) → valuta su GT
   ```

---