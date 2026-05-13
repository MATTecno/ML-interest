"""
API publica do modelo.

Este modulo preserva as chamadas antigas (`import model as mdl`) enquanto a
implementacao fica separada em dataset.py, model_training.py e predictor.py.
"""

from dataset import (
    CSV_FIELDNAMES,
    PROFILES_PATH,
    SYNTHETIC_PATH,
    _ensure_profiles_schema,
    _load_csv,
    load_all_data,
    save_labeled_profile,
    count_real_profiles,
    count_trainable_real_profiles,
    should_retrain,
)
from features import load_config
from model_training import MODEL_PATH, TRAINING_VERSION, load_model, train_model
from predictor import predict


__all__ = [
    "CSV_FIELDNAMES",
    "MODEL_PATH",
    "TRAINING_VERSION",
    "PROFILES_PATH",
    "SYNTHETIC_PATH",
    "_ensure_profiles_schema",
    "_load_csv",
    "load_all_data",
    "save_labeled_profile",
    "count_real_profiles",
    "count_trainable_real_profiles",
    "should_retrain",
    "load_config",
    "load_model",
    "train_model",
    "predict",
]
