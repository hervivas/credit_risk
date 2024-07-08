import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def set_table_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    for col in df.columns:
        if col[-1] in ("P", "A"):
            df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
        if col[-1] in ("D"):
            df = df.with_columns(pl.col(col).cast(pl.Date).alias(col))
    return df

def convert_strings(df: pl.DataFrame) -> pl.DataFrame:
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(pl.col(col).cast(pl.Categorical))
    return df

def evaluate_model(model, X_test, y_test):
    # Get probabilities
    y_proba = model.predict_proba(X_test)
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])

    # Calculate the best threshold
    distances = [euclidean((0, 1), (fpr_i, tpr_i)) for fpr_i, tpr_i in zip(fpr, tpr)]
    best_idx = np.argmin(distances)
    best_thresh = thresholds[best_idx]
      
    # Plot the ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='Random classification')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Evaluate metrics using the best threshold
    y_pred_best = (y_proba[:, 1] >= best_thresh).astype(int)

    accuracy_best = accuracy_score(y_test, y_pred_best)
    precision_best = precision_score(y_test, y_pred_best)
    recall_best = recall_score(y_test, y_pred_best)
    f1_best = f1_score(y_test, y_pred_best)

    print("Best threshold:", best_thresh)
    print("\nMetrics using best threshold:")
    print("Accuracy:", accuracy_best)
    print("Precision:", precision_best)
    print("Recall:", recall_best)
    print("F1-score:", f1_best)
