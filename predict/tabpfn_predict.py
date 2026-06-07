import numpy as np
import torch

from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.many_class import ManyClassClassifier


FEATURE_LIMIT = 500
SAMPLE_LIMIT = 10_000


def _ensure_numpy(array_like):
    """Convert tensors/lists to numpy arrays on CPU."""
    if torch.is_tensor(array_like):
        return array_like.detach().cpu().numpy()
    if isinstance(array_like, np.ndarray):
        return array_like
    return np.asarray(array_like)


def tabpfn_predict(X_train, y_train, X_test, device):
    """
    Train a TabPFN classifier and return predictions for X_test.

    Returns either class probabilities (if the estimator supports them) or
    hard class predictions as a numpy array.
    """

    X_train_np = _ensure_numpy(X_train).astype(np.float32)
    X_test_np = _ensure_numpy(X_test).astype(np.float32)
    y_train_np = _ensure_numpy(y_train).reshape(-1)

    # Encode labels to consecutive integers so downstream consumers can map
    # predictions back to the original labels reliably.
    unique_classes = np.unique(y_train_np)
    class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
    encoded_y = np.asarray([class_to_index[cls] for cls in y_train_np], dtype=np.int64)

    feature_dim = X_train_np.shape[1] if X_train_np.ndim == 2 else 0
    total_samples = X_train_np.shape[0] + X_test_np.shape[0]
    over_feature_limit = feature_dim > FEATURE_LIMIT
    over_sample_limit = total_samples > SAMPLE_LIMIT

    tabpfn_kwargs = {"device": device}
    if over_feature_limit or over_sample_limit:
        tabpfn_kwargs["ignore_pretraining_limits"] = True

    base_estimator = TabPFNClassifier(**tabpfn_kwargs)

    is_many_class = len(unique_classes) > 10
    if is_many_class:
        classifier = ManyClassClassifier(
            estimator=base_estimator,
            alphabet_size=10,
            n_estimators=None,
            n_estimators_redundancy=4,
        )
    else:
        classifier = base_estimator

    classifier.fit(X_train_np, encoded_y)

    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(X_test_np)
        if torch.is_tensor(proba):
            proba = proba.detach().cpu().numpy()
        # Reorder columns to align with the sorted unique classes.
        proba = np.asarray(proba)
        if proba.shape[1] != len(unique_classes):
            raise RuntimeError("TabPFN returned probabilities with unexpected shape.")
        return proba

    preds = classifier.predict(X_test_np)
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    preds = np.asarray(preds, dtype=np.int64)

    # Delete to free memory
    del classifier
    
    return preds
