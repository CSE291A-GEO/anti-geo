"""
Classification Module for GEO Detection

This module provides machine learning classifiers for GEO detection,
including SVM-based classification trained on semantic matching features.
"""

from .svm_classifier import SVMGEODetector, train_svm_classifier

__all__ = ['SVMGEODetector', 'train_svm_classifier']

