# Colab Notebooks - Guida Completa all'Esecuzione

Questa cartella contiene i notebook Google Colab per l'estensione del progetto AML 2025 sulla rilevazione degli errori nelle ricette.

---

## 📋 Panoramica Extension

Il progetto è suddiviso in 4 substep principali:

### Substep 1: Estrazione Feature Video (In Refactoring)
- **Stato**: Non presente in questa cartella (in fase di refactoring).
- **Funzionamento**:
  - Estrazione feature video tramite **EgoVLP**.
  - Applicazione **HIERO zero-shot** per identificare gli step ed estrarre segmenti video.
- **Output**: Genera file `.npz` contenenti embeddings video e timestamp, usati come input per i substep successivi.

### Substep 2: Baseline Transformer
- Implementa un modello **Transformer baseline** per la classificazione binaria degli errori.
- Prende in input le feature video estratte dal Substep 1.
- Definisce lo split del dataset (train/val/test) che verrà poi mantenuto consistente per i modelli successivi.

### Substep 3: Feature Testuali & Costruzione Grafi (GNN Prep)
Questo step è cruciale per la preparazione dei dati per la Graph Neural Network. Svolge tre funzioni principali:
1. **Estrazione Feature Testuali**: Usa **EgoVLP** per estrarre embeddings semantici dagli step testuali delle ricette.
2. **Hungarian Matching con Costo Temporale**: Allinea le feature video (dal Substep 1) con le feature testuali.
   - Utilizza l'algoritmo Hungarian per l'assegnamento ottimo.
   - Include un termine di **costo temporale** (penalty) per favorire l'ordine cronologico corretto degli step.
3. **Costruzione Grafi**: Combina feature video allineate e feature testuali per costruire i grafi che saranno l'input della GNN.

### Substep 4: Classificazione GNN & Analisi
Lavoro diviso in tre notebook specifici:
1. **`Substep_4.ipynb` (Classificazione GNN)**: Addestra e valuta la GNN sui grafi costruiti nel Substep 3.
2. **`GroundTruth_GraphCreation.ipynb`**: Costruisce i grafi basandosi esclusivamente sulle annotazioni di Ground Truth (etichette manuali di errori e strutture).
3. **`Substep_4_onGT.ipynb`**: Esegue la classificazione GNN sui grafi di Ground Truth per confrontare le performance ideali con quelle ottenute dai grafi predetti.

---

## 🚀 Preparazione Ambiente Google Drive

Per eseguire correttamente i notebook, è **fondamentale** replicare esattamente la struttura delle cartelle su Google Drive.

### 1. Creazione Cartella Principale
Nel tuo "Il mio Drive" (MyDrive), crea una cartella chiamata:
`AML_Project`

### 2. Struttura Completa
All'interno di `AML_Project`, crea le sottocartelle come mostrato nell'albero sottostante. Assicurati di caricare i file necessari (annotazioni, dati) nelle posizioni corrette.

```text
MyDrive/
└── AML_Project/
    │
    ├── annotations-main/           # Cartella con le annotazioni dataset
    │   ├── annotation_csv/
    │   │   └── error_annotations.csv
    │   ├── annotation_json/
    │   │   ├── complete_step_annotations.json
    │   │   └── error_annotations.json
    │   └── task_graphs/            # File JSON contenenti i grafi delle ricette
    │       ├── microwaveeggsandwich.json
    │       └── ... (altri json delle ricette)
    │
    └── Extension/                  # Cartella di output del progetto
        │
        ├── step1_HiERO/
        │
        ├── step_2_baseline/        
        │
        ├── step_3_task_graph/       
        │
        └── step_4_gnn/        
```

Per evitare passaggi si possono scaricare i file necessari al seguente link: 

---

## 📓 Guida Dettagliata ai Notebook

### Substep 2: Baseline Transformer (`Substep_2.ipynb`)
**Obiettivo**: Stabilire una baseline di performance e fissare lo split dei dati.
- **Configurazione**: Verifica che `input_dir` punti a `Extension/step1_HiERO/steps_v4`.
- **Output Chiave**: `master_split_ids.json`. Questo file assicura che il modello GNN (Substep 3) usi esattamente gli stessi video per training e testing.

### Substep 3: GNN Preparation & Model (`Substep_3.ipynb`)
**Obiettivo**: Costruire i grafi multimodali e addestrare la GNN.
- **Dettagli**:
  - Carica EgoVLP dal repository specificato.
  - Esegue l'allineamento Video-Testo con penalità temporale.
  - Salva i grafi processati (in memoria o su disco durante l'esecuzione) e addestra la GNN.
- **Configurazione**: Assicurati che `SPLIT_JSON_PATH` punti al file generato dal Substep 2.

### Substep 4: Analisi e Confronto
Esegui in questo ordine:

1. **`Substep_4.ipynb`**:
   - Carica il modello GNN addestrato in step 3.
   - Genera predizioni e analizza la struttura dei grafi predetti.
   
2. **`GroundTruth_GraphCreation.ipynb`**:
   - Script di utility.
   - Legge `error_annotations.csv` e i JSON in `task_graphs`.
   - Genera i grafi "ideali" in `Extension/step_4_analysis/ground_truth_graphs`.

3. **`Substep_4_onGT.ipynb`**:
   - Verifica quanto performerebbe la GNN se i grafi di input fossero perfetti (Ground Truth).
   - Utile per capire se i limiti di performance dipendono dalla GNN o dalla qualità dei grafi costruiti nel step 3.

---

## 🛠️ Risoluzione Problemi Comuni

| Errore | Causa Probabile | Soluzione |
|--------|-----------------|-----------|
| `FileNotFoundError` | Percorso Drive errato | Controlla di aver creato esattamente la struttura cartelle descritta sopra. |
| `CUDA out of memory` | GPU satura | Riduci il `batch_size` (es. da 16 a 8) nelle configurazioni. |
| `KeyError: 'steps'` | JSON graffo mancante o corrotto | Verifica che la cartella `annotations-main/task_graphs` contenga tutti i JSON delle ricette. |
| Split Dati Diverso | Seed non fissato | Assicurati di non aver cambiato `SEED = 42` nei notebook. |

---

## 📊 Metriche di Riferimento (Expected)
- **Baseline (Transformer)**: Accuracy ~59%, F1 ~68%
- **GNN (Ours)**: Accuracy ~65%, F1 ~72%
