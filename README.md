# Multimodal Misinformation Detection

A machine learning pipeline that classifies fake vs real content using both images and text together, built with OpenAI's CLIP (Contrastive Language-Image Pretraining) — a Vision-Language Transformer model.

## Overview

Misinformation increasingly spreads as image+text combinations — memes, screenshots, sensationalized photos with misleading captions. Text-only classifiers miss this signal entirely. This project uses CLIP to jointly represent both modalities and trains a classifier on top of the combined features.

## Dataset

[Fakeddit](https://www.kaggle.com/datasets/vanshikavmittal/fakeddit-dataset) — a large-scale multimodal fake news dataset built from Reddit posts, containing images, titles, and fake/real labels.

A random sample of ~2,000 image-text pairs was used, with ~1,900 successfully downloaded after filtering broken image links.

## Pipeline

1. **Data sampling** — filtered to rows with valid images, sampled 2,000 pairs
2. **Image download** — fetched actual images from hosted URLs
3. **Feature extraction** — used pretrained CLIP (ViT-B/32) to convert both images and text into 512-dimensional vectors each
4. **Feature fusion** — concatenated image + text vectors into a single 1024-dimensional representation per sample
5. **Classification** — trained a Logistic Regression classifier on top of CLIP's frozen features
6. **Evaluation** — tested on a held-out 20% split

## Results

- **Accuracy**: 86%
- **Fake precision/recall**: 0.81 / 0.83
- **Real precision/recall**: 0.89 / 0.88

## Limitations

- Dataset is noisier and more diverse than curated news datasets (memes, screenshots, general Reddit content), making this a harder task than text-only classification
- The classifier appears to weight image features more heavily than caption text, especially for out-of-distribution images — sensational caption changes did not always flip predictions
- Class imbalance in training data (more "real" examples) likely biases the model toward predicting "real" under uncertainty
- CLIP features were used frozen (not fine-tuned); fine-tuning could likely improve results further

## Tech Stack

- Python, PyTorch
- OpenAI CLIP (ViT-B/32)
- scikit-learn (Logistic Regression)

## Future Work

- Fine-tune CLIP end-to-end rather than using frozen features
- Address class imbalance (weighted loss or oversampling)
- Compare against a text-only and image-only baseline to quantify the actual benefit of multimodal fusion

## Author

Fazia Akhtar
