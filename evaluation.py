import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from train import y_test, lr_pred


# ---------------- TOP CLASSES ----------------
top_classes = y_test.value_counts().head(10).index

mask = y_test.isin(top_classes)

y_true_top = y_test[mask]
y_pred_top = pd.Series(lr_pred, index=y_test.index)[mask]

# ---------------- CLASSIFICATION REPORT ----------------
print("\n===== Classification Report (Top Classes Only) =====\n")
print(classification_report(y_true_top, y_pred_top))

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true_top, y_pred_top, labels=top_classes)

plt.figure(figsize=(10, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=top_classes,
    yticklabels=top_classes
)

plt.title("Confusion Matrix - Top 10 Classes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()