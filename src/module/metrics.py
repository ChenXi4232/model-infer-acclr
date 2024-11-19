import numpy as np
from typing import Optional
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

def acc(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weight: bool = False,
        weights: Optional[np.ndarray] = None,
    ) -> float:
    """
    Compute the accuracy of the model.

    Args:
        y_true: np.ndarray
            The true labels.
        y_pred: np.ndarray
            The predicted labels.
        weights: np.ndarray
            The weights to apply to the accuracy.

    Returns:
        float
            The accuracy of the model.
    """
    if weight:
        use_weights = np.zeros_like(y_true, dtype=float)
        if weights is None:
            # weighted by sample distribution
            unique_classes, counts = np.unique(y_true, return_counts=True)
            class_weights = compute_class_weight("balanced", classes=np.unique(y_true), y=y_true)
            for i, label in enumerate(unique_classes):
                use_weights[y_true == label] = class_weights[i]
            
        return accuracy_score(y_true, y_pred, sample_weight=use_weights)


    return accuracy_score(y_true, y_pred)