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
MyDrive/
└── AML_Project/                          # CARTELLA PRINCIPALE
    │
    ├── annotations-main/                  # Dati di Annotazione e Grafi delle Ricette
    │   ├── annotation_csv/
    │   │   └── error_annotations.csv     # CSV con etichette di errore (recording_id, is_error)
    │   ├── annotation_json/
    │   │   ├── complete_step_annotations.json   # Annotazioni complete: timing degli step
    │   │   ├── error_annotations.json           # JSON con etichette di errore
    │   │   ├── environment_combined_splits.json # Split per ambiente
    │   │   ├── person_combined_splits.json      # Split per persona
    │   │   └── recordings_combined_splits.json  # Split per video
    │   └── task_graphs/                  # Grafi canonici delle ricette (JSON)
    │       ├── microwaveeggsandwich.json
    │       ├── blenderbananapancakes.json
    │       └── ... (un JSON per ogni ricetta)
    |
    │__ First_Part/
    |   |__ features/
    │   |   │__ omnivore.zip
    |   |   |__ slowfast.zip
    |   |__ models_result_omnivore/
    |   |   |__ lstm_step/
    |   |   |__ lstm_recordings/
    |   |__ models_result_slowfast/
    |   |__ error_recognition_best.zip
    │
    ├── 3_EgoVLP/                          # Modello EgoVLP (Feature Extractor)
    │   ├── EgoVLP-main/                   # Repository EgoVLP clonato
    │   │   ├── model/
    │   │   ├── utils/
    │   │   └── ...
    │   ├── checkpoints/
    │   │   └── egovlp.pth                 # Checkpoint pre-addestrato EgoVLP
    │   └── features/                      # Feature video estratte (output Substep 1)
    │   |   └── {recording_id}_360p_224.mp4_1s_1s.npy  # Embeddings video (dim: [T, 256])
    |   |__ videos/
    |       |__ 1_7_360p_224.mp4
    |       |__ ...
    |__ pretrained
    |   |__ jx_vit_base_p16_224-80ecf9dd.pth
    │
    └── Extension/                         # CARTELLA OUTPUT: Risultati dei 4 Substep
        │
        ├── step_1_HiERO/                  # [SUBSTEP 1] Estrazione Feature Video
        │   └── steps/
        │       └── {recording_id}_steps.npz  # Video embeddings + timestamp di ogni step
        │           Contiene:
        │           - 'embeddings': [N_steps, 256] (feature video per step)
        │           - 'segments': [N_steps, 2] (start_time, end_time di ogni step)
        │
        ├── step_2_baseline/                # [SUBSTEP 2] Baseline Transformer
        │   └── model_result/
        │       └── master_split_ids.json   # Split FIXED: {'train': [...], 'val': [...], 'test': [...]}
        │                                   # CRUCIALE: Questo file è replicato in tutti gli step successivi
        │
        ├── step_3_task_graph/              # [SUBSTEP 3] Feature Testuali e Grafi
        │   ├── text_features_egovlp/       # Feature testuali estratte (output intermedio)
        │   │   └── {recipe_id}.pt          # Embeddings testuali: [N_steps, 256]
        │   ├── matched_features/           # Allineamento Video-Testo (output intermedio)
        │   │   └── match_{recording_id}.pt # Contiene:
        │   │       - 'video_features': [N_video_steps, 256]
        │   │       - 'text_features': [N_recipe_steps, 256]
        │   │       - 'matches': [(video_idx, recipe_idx), ...]  # Risultato Hungarian
        │   │       - 'cost_matrix': [N_video_steps, N_recipe_steps]
        │   └── gnn_ready_data/             # Grafi per GNN (input Substep 4)
        │       └── gnn_ready_{recording_id}.pt  # Grafo costruito da predizioni:
        │           Contiene:
        │           - 'x_text': [N_steps, 256] (feature testuali)
        │           - 'x_video': [N_steps, 256] (feature video allineate)
        │           - 'edge_index': [2, E] (archi dal grafo della ricetta)
        │           - 'y': 0 o 1 (etichetta: errore o no)
        │           - 'vid_id': recording_id
        │           - 'recipe': nome ricetta
        │
        └── step_4_gnn/                     # [SUBSTEP 4] GNN Classification & Analysis
            ├── gnn_ready_data_groundtruth/  # Grafi costruiti da Ground Truth
            │   └── gnn_ready_gt_{recording_id}.pt  # Come gnn_ready_data ma con timing GT
            ├── models/
            │   └── best_gnn_model.pth      # Modello GNN addestrato
            └── results/
                ├── confusion_matrix.png
                └── metrics.txt
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