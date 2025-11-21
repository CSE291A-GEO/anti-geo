#!/bin/bash
# Run all models with and without semantic features

cd /Users/xurui/Downloads/FA25/291A_AI_Systems/anti-geo

echo "=========================================="
echo "Running All Model Experiments"
echo "=========================================="

# SVM - Without semantic features
echo "Running SVM without semantic features..."
conda run -n fifteenAI python src/classification/svm_classifier.py \
    --opt-data optimization_dataset.json \
    --limit 1000 --train-size 300 --validation-size 700 \
    --C 1.5 \
    --output src/classification \
    --model-name svm_baseline.pkl 2>&1 | tee logs/svm_baseline.log

# SVM - With semantic features
echo "Running SVM with semantic features..."
conda run -n fifteenAI python src/classification/svm_classifier.py \
    --opt-data optimization_dataset.json \
    --limit 1000 --train-size 300 --validation-size 700 \
    --C 1.5 --use-semantic-features \
    --output src/classification \
    --model-name svm_with_semantic.pkl 2>&1 | tee logs/svm_semantic.log

# Logistic Ordinal - Without semantic features
echo "Running Logistic Ordinal without semantic features..."
conda run -n fifteenAI python src/classification/logistic_ordinal_classifier.py \
    --opt-data optimization_dataset.json \
    --limit 1000 --train-size 300 --validation-size 700 \
    --C 1.0 \
    --output src/classification \
    --model-name logistic_baseline.pkl 2>&1 | tee logs/logistic_baseline.log

# Logistic Ordinal - With semantic features
echo "Running Logistic Ordinal with semantic features..."
conda run -n fifteenAI python src/classification/logistic_ordinal_classifier.py \
    --opt-data optimization_dataset.json \
    --limit 1000 --train-size 300 --validation-size 700 \
    --C 1.0 --use-semantic-features \
    --output src/classification \
    --model-name logistic_with_semantic.pkl 2>&1 | tee logs/logistic_semantic.log

# GBM - Without semantic features
echo "Running GBM without semantic features..."
conda run -n fifteenAI python src/classification/gbm_ordinal_classifier.py \
    --opt-data optimization_dataset.json \
    --limit 1000 --train-size 300 --validation-size 700 \
    --n-estimators 100 --learning-rate 0.1 --max-depth 3 \
    --output src/classification \
    --model-name gbm_baseline.pkl 2>&1 | tee logs/gbm_baseline.log

# GBM - With semantic features
echo "Running GBM with semantic features..."
conda run -n fifteenAI python src/classification/gbm_ordinal_classifier.py \
    --opt-data optimization_dataset.json \
    --limit 1000 --train-size 300 --validation-size 700 \
    --n-estimators 100 --learning-rate 0.1 --max-depth 3 --use-semantic-features \
    --output src/classification \
    --model-name gbm_with_semantic.pkl 2>&1 | tee logs/gbm_semantic.log

# Neural - Without semantic features
echo "Running Neural without semantic features..."
conda run -n fifteenAI python src/classification/neural_ordinal_classifier.py \
    --opt-data optimization_dataset.json \
    --limit 1000 --train-size 300 --validation-size 700 \
    --hidden-dim 128 --learning-rate 0.001 --epochs 50 --batch-size 32 --num-layers 10 \
    --output src/classification \
    --model-name neural_baseline.pkl 2>&1 | tee logs/neural_baseline.log

# Neural - With semantic features
echo "Running Neural with semantic features..."
conda run -n fifteenAI python src/classification/neural_ordinal_classifier.py \
    --opt-data optimization_dataset.json \
    --limit 1000 --train-size 300 --validation-size 700 \
    --hidden-dim 128 --learning-rate 0.001 --epochs 50 --batch-size 32 --num-layers 10 --use-semantic-features \
    --output src/classification \
    --model-name neural_with_semantic.pkl 2>&1 | tee logs/neural_semantic.log

# RNN - Without semantic features
echo "Running RNN without semantic features..."
conda run -n fifteenAI python src/classification/rnn_ordinal_classifier.py \
    --opt-data optimization_dataset.json \
    --limit 1000 --train-size 300 --validation-size 700 \
    --rnn-type GRU --num-layers 3 --bidirectional \
    --hidden-dim 128 --learning-rate 0.001 --epochs 50 --batch-size 32 \
    --output src/classification \
    --model-name rnn_baseline.pkl 2>&1 | tee logs/rnn_baseline.log

# RNN - With semantic features
echo "Running RNN with semantic features..."
conda run -n fifteenAI python src/classification/rnn_ordinal_classifier.py \
    --opt-data optimization_dataset.json \
    --limit 1000 --train-size 300 --validation-size 700 \
    --rnn-type GRU --num-layers 3 --bidirectional --use-semantic-features \
    --hidden-dim 128 --learning-rate 0.001 --epochs 50 --batch-size 32 \
    --output src/classification \
    --model-name rnn_with_semantic.pkl 2>&1 | tee logs/rnn_semantic.log

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

