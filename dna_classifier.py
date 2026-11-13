# Predict the Genome Pattern: ML classification of DNA sequences
# Works in Google Colab (each "# %%" block = one cell) or as a normal script.

# %% CELL 1 - Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score

# %% CELL 2 - Dataset (Sequence ID, DNA Sequence, Group)
data = [
    ("S1", "ATGCGATC", "A"),
    ("S2", "ATGCGTAC", "A"),
    ("S3", "GCGCGTAT", "B"),
    ("S4", "GCGCGCGC", "B"),
]
df = pd.DataFrame(data, columns=["Sequence_ID", "DNA_Sequence", "Group"])

# %% CELL 3 - Feature extraction
def extract_features(seq):
    seq = seq.upper()
    length = len(seq)
    a, t, g, c = seq.count("A"), seq.count("T"), seq.count("G"), seq.count("C")
    gc = round((g + c) / length * 100, 2)
    return {"A": a, "T": t, "G": g, "C": c, "Length": length, "GC_percent": gc}

features = pd.DataFrame([extract_features(s) for s in df["DNA_Sequence"]])
table = pd.concat([df, features], axis=1)
print("FEATURE TABLE")
print(table.to_string(index=False))

# %% CELL 4 - Train the model
FEATURE_COLS = ["A", "T", "G", "C", "Length", "GC_percent"]
X = table[FEATURE_COLS]
y = table["Group"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

train_acc = accuracy_score(y, model.predict(X))
print(f"\nTraining accuracy: {train_acc * 100:.1f}%")

# With only 4 sequences a train/test split is not meaningful,
# so we use Leave-One-Out cross-validation as an honest estimate.
loo_acc = cross_val_score(DecisionTreeClassifier(random_state=42), X, y,
                          cv=LeaveOneOut()).mean()
print(f"Leave-One-Out accuracy: {loo_acc * 100:.1f}%")

# %% CELL 5 - Feature importance
importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
print("\nFEATURE IMPORTANCE")
print(importance.sort_values(ascending=False).round(3).to_string())

importance.sort_values().plot(kind="barh", title="Feature importance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plot_tree(model, feature_names=FEATURE_COLS, class_names=list(model.classes_),
          filled=True)
plt.title("Decision Tree")
plt.show()

# %% CELL 6 - Predict the unknown sequence
unknown = "ATGCGCGATC"
unk_features = pd.DataFrame([extract_features(unknown)])[FEATURE_COLS]
print("\nUNKNOWN SEQUENCE FEATURES")
print(unk_features.to_string(index=False))

pred = model.predict(unk_features)[0]
proba = model.predict_proba(unk_features)[0]
print(f"\nPredicted group for {unknown}: Group {pred}")
print("Probabilities:", dict(zip(model.classes_, proba.round(2))))

# %% CELL 7 - Critical check: is the prediction stable?
# With only 4 training sequences several features separate the groups equally
# well (A count, G count, GC%), so the tree can pick any of them.
print("\nSTABILITY CHECK (which features the model is allowed to use)")
for name, cols in [("All 6 features", FEATURE_COLS),
                   ("GC_percent only", ["GC_percent"]),
                   ("A count only", ["A"])]:
    m = DecisionTreeClassifier(random_state=42).fit(table[cols], y)
    p = m.predict(pd.DataFrame([extract_features(unknown)])[cols])[0]
    cv = cross_val_score(DecisionTreeClassifier(random_state=42), table[cols], y,
                         cv=LeaveOneOut()).mean()
    print(f"{name:18s} -> predicts Group {p} | LOO accuracy {cv*100:.0f}%")
