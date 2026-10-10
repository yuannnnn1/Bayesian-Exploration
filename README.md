# Bayesian Exploration

This repository implements a small, reproducible version of **Bayesian Exploration of Pre-trained Models for Low-shot Image Classification** (Miao et al., CVPR 2024).

The assignment report is [REPORT.md](REPORT.md). Actual experimental evidence, logs, and figures are in [results/](results/).

The implementation follows the paper's core design: frozen pretrained image encoders define a sum of RBF deep kernels, a scaled CLIP text-image softmax provides the prior mean, and exact Gaussian-process label regression produces posterior predictions and latent variance. Oxford-IIIT Pet is a manageable local benchmark that is also evaluated in the original paper. This is a reduced implementation with two encoders and a different validation protocol, rather than a complete reproduction of all published experiments.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-experiment.txt
```

## Run

```bash
.venv/bin/python prepare_features.py
.venv/bin/python run_experiment.py
.venv/bin/python analyze_results.py
```

The first command downloads Oxford-IIIT Pet, OpenCLIP RN50 (OpenAI), and DINO ResNet-50 weights into ignored paths under data/ and the user's model cache. QuickGELU is explicitly enabled for compatibility with the CLIP weights. Features and per-image predictions are stored under artifacts/. Portable experiment evidence is saved under results/: experiment.json, summary.csv, experiment.log, validation.json, and three PNG figures.

The local implementation uses CPU because MPS is unavailable on the current machine. Feature preparation has been verified with 3,680 trainval images and 3,669 test images. The cached tensors contain normalized OpenCLIP image features, normalized DINO ResNet-50 features, Oxford-IIIT Pet category labels, and normalized CLIP text features for the 37 class names.

If image features are already cached, only the text metadata needs to be refreshed after changing the CLIP text-encoder configuration:

    .venv/bin/python prepare_features.py --text-only

## Experiment Protocol

- 37 classes; 3,680 trainval images and 3,669 official test images.
- 1, 2, 4, 8, and 16 labeled images per class; seeds 0, 1, and 2. Supports are nested across shot counts for each seed.
- All methods use the same supports. Hyperparameter selection receives no additional labels.
- For shots greater than 1, each class's support is split 1:1 into fit/validation. GP parameters optimize the full joint Gaussian predictive likelihood; the ridge probe selects its regularization by validation MSE. The final predictor uses all support samples.
- For 1-shot, GP parameters optimize marginal likelihood; ridge regularization is fixed at 0.01.
- GP optimization: 100 Adam updates, learning rate 0.01 with cosine decay. Each encoder has a scalar lengthscale. Initial noise variance is 0.01, CLIP temperature is 100, and prior scale is 1.
- Comparisons: zero-shot CLIP, a CLIP ridge label-regression probe, a CLIP-only GP ablation, and the CLIP+DINO GP. This probe is not the paper's Ens-LP ensemble baseline.
- Accuracy and macro-F1 are computed on all test images. Means and sample standard deviations are computed across three support seeds; the deterministic zero-shot baseline is evaluated once.
- GP/ridge categorical NLL and 10-bin ECE use regression scores clipped to a positive floor and row-normalized. These probabilities are a reporting convention, not the exact Gaussian predictive likelihood or the paper's raw-mean confidence definition. Classification still uses the same argmax as the raw GP mean.
- Uncertainty is the diagonal of the latent posterior covariance, not class-specific probability variance. Selective prediction is examined on the in-distribution test set; no OOD experiment is claimed.
- Recorded run time excludes model download and image feature extraction. It includes support-level tuning and final feature-based prediction; the first GP run includes optimizer initialization overhead.

The evidence records support/validation indices, all optimization losses, fitted hyperparameters, source/cache/prediction SHA-256 hashes, and runtime. The analysis command recomputes metrics from saved predictions, checks the splits, and generates figures without refitting.

## Verification

    .venv/bin/python -m unittest discover -s tests -v

Seven focused tests cover posterior formulas, full-matrix versus diagonal variance, Gaussian likelihood, kernel construction, prior/probability mapping, both tuning objectives, stratified sampling, and macro-F1.

## References

- Paper: https://arxiv.org/html/2404.00312v1
- CVPR publication: https://openaccess.thecvf.com/content/CVPR2024/html/Miao_Bayesian_Exploration_of_Pre-trained_Models_for_Low-shot_Image_Classification_CVPR_2024_paper.html
- OpenCLIP: https://github.com/mlfoundations/open_clip
- DINO: https://github.com/facebookresearch/dino
- Assignment source repository: https://github.com/yuannnnn1/Bayesian-Exploration
