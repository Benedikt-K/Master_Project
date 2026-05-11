"""Data loading and splitting module for CRISPR array examples.

Provides stratified splitting strategies and PyTorch dataset wrappers.
"""

from .loading import (
    DirectionTorchDataset,
    batch_to_tensors,
    build_dataloader,
    collate_for_training,
)
from .splitting import (
    build_cv_folds_by_signature,
    split_dev_pool_by_mode,
    split_groups,
    stratified_holdout_by_mode,
    stratified_split_by_cas_subtype,
    stratified_train_test_and_val_by_cas_subtype,
    stratified_train_test_and_val_by_cas_subtype_and_label,
    stratified_train_test_and_val_by_label,
    stratified_train_test_by_cas_subtype,
)

__all__ = [
    "split_groups",
    "stratified_holdout_by_mode",
    "stratified_split_by_cas_subtype",
    "stratified_train_test_by_cas_subtype",
    "stratified_train_test_and_val_by_cas_subtype",
    "stratified_train_test_and_val_by_label",
    "stratified_train_test_and_val_by_cas_subtype_and_label",
    "split_dev_pool_by_mode",
    "build_cv_folds_by_signature",
    "DirectionTorchDataset",
    "batch_to_tensors",
    "collate_for_training",
    "build_dataloader",
]
