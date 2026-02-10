## Problem Statement

Relation extraction is formulated as a multi-class classification task. Given a sentence containing two marked entities (<e1>, <e2>), the objective is to predict the semantic relation holding between them. The dataset defines 19 relation classes, including directionality and a generic Other class.

While pretrained language models capture rich contextual semantics, they do not explicitly encode sentence-level syntactic structure. This project evaluates whether dependency-based graph representations can complement transformer embeddings in this setting.

---

## Dataset

The experiments use the SemEval-2010 Task 8 dataset, a standard benchmark for relation classification.

- Training samples: 8,000
- Test samples: 2,717
- Relation labels: 19 (18 directed relations + Other)

The dataset is downloaded automatically from a public mirror if not present locally, ensuring reproducibility.

---

## Methodology

### Baselines

1. **Majority Class Baseline**
   A simple frequency-based predictor that always selects the most common relation. This establishes a lower-bound performance reference.

2. **DistilBERT Classifier**
   A strong neural baseline built on DistilBERT:
   - Uses the [CLS] token embedding as a sentence-level representation
   - Two-layer feedforward classification head
   - Class-weighted cross entropy to mitigate label imbalance

This model serves as a competitive transformer-only reference.

---

### Proposed Model: DistilBERT + Graph Attention Network

The core contribution of this project is a hybrid architecture that integrates syntactic structure into transformer-based representations.

Pipeline:
- Sentences are parsed using spaCy to obtain dependency trees
- Each sentence is converted into a directed graph:
  - Nodes represent tokens
  - Edges represent dependency relations (bidirectional)
- DistilBERT token embeddings initialize node features
- Two stacked GAT layers propagate contextual information across the graph
- Entity-specific node representations are pooled and classified

This design allows the model to jointly reason over contextual semantics and explicit dependency structure.

---

## Project Structure

- `app.py`
  - End-to-end experiment pipeline
  - Data ingestion, preprocessing, graph construction
  - Model definitions, training loops, evaluation, visualization

- `infer.py`
  - Lightweight inference utility for loading trained checkpoints and running predictions on new sentences

- `GAT.ipynb`
  - Detailed project ipynb covering motivation, design decisions, experiments, and analysis

---

## Quick Start (Minimal)

### Environment Setup

Python 3.9+ is recommended. GPU acceleration is optional but encouraged.

Install core dependencies:

```
pip install torch transformers torch-geometric spacy scikit-learn pandas numpy matplotlib seaborn
python -m spacy download en_core_web_sm
```

---

### Train and Evaluate

Run the full pipeline:

```
python app.py
```

This will:
- Download the dataset if necessary
- Train the DistilBERT baseline
- Tune GAT hyperparameters
- Train the final hybrid model
- Evaluate all models on the test set
- Generate metrics, plots, and analysis artifacts

---

### Inference

After training :

```
python infer.py
```

This loads the trained model and runs inference on custom input sentences.

---

## Evaluation Metrics

Models are evaluated using:
- Accuracy
- Macro F1 score (primary metric)
- Weighted F1 score

Macro F1 is emphasized due to significant class imbalance across relation types.

---

## Results

On the SemEval-2010 Task 8 test set:

- Majority baseline establishes a low performance floor
- DistilBERT baseline achieves strong performance (Macro F1 ≈ 0.69)
- DistilBERT + GAT further improves performance (Macro F1 ≈ 0.76)

The hybrid model demonstrates consistent gains across multiple relation categories, particularly for structurally dependent relations and longer entity distances.

---

## Analysis and Visualization

The project includes extensive analysis tools:
- Relation label distribution and sentence statistics
- Training and validation loss curves
- Confusion matrix and per-class F1 comparisons
- t-SNE visualization of learned representations
- Performance breakdown by entity distance

These diagnostics are intended to surface both strengths and failure modes of the proposed approach.

---

## License

This project is released for research and educational use. Refer to the SemEval dataset license and pretrained model licenses for any downstream usage constraints.

---

## Acknowledgements

- SemEval-2010 Task 8
- Hugging Face Transformers
- PyTorch Geometric
- spaCy