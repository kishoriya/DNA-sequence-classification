# Predict the Genome Pattern: ML Classification of DNA Sequences

Course: 26BTB101T – Introduction to Computational Biology (B.Tech Biotechnology, Sem I)

A Decision Tree classifier that learns to separate DNA sequences into Group A and Group B
from simple numeric features, then predicts the group of an unknown sequence (`ATGCGCGATC`).

## Features extracted
A, T, G, C counts, sequence length, GC content (%).

## Dataset
| ID | Sequence | Group |
|----|----------|-------|
| S1 | ATGCGATC | A |
| S2 | ATGCGTAC | A |
| S3 | GCGCGTAT | B |
| S4 | GCGCGCGC | B |

## Run
**Google Colab:** open `dna_classifier.ipynb` via File → Upload notebook, then Runtime → Run all.

**Locally:**
```bash
pip install -r requirements.txt
python dna_classifier.py
```

## Results
- Training accuracy: 100%; Leave-One-Out accuracy: 100% (all 6 features)
- Unknown sequence `ATGCGCGATC` (A=2, T=2, G=3, C=3, Length=10, GC=60%): **Group A** with all 6 features
- Stability check: using GC% only predicts **Group B** (LOO accuracy 75%)

## Key takeaway
With only 4 training sequences, several features (A count, G count, GC%) separate the groups
equally well, so the prediction depends on which feature the tree picks. Training length was
constant (8), so length carries no information. Larger, more diverse datasets are needed.

## Limitations of AI/ML here
Tiny training set → overfitting; no "unknown" option for out-of-distribution sequences; limited interpretability.

SDG alignment: SDG 3 (Good Health), SDG 9.5 (Research & Innovation).
