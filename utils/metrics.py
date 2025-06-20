from sklearn.metrics import (
  accuracy_score, precision_score, recall_score,
  f1_score, confusion_matrix, classification_report
)
import numpy as np

def calculate_metrics(y_true, y_pred, average='macro'):
  return {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
    "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
    "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0)
  }

def print_classification_report(y_true, y_pred, target_names=None):
  report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
  print(report)

def get_confusion_matrix(y_true, y_pred, labels=None):
  return confusion_matrix(y_true, y_pred, labels=labels)
