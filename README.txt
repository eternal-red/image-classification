Overview
This notebook trains a CNN classifier on 8631 face identities and uses the trained model as an embedding extractor for pairwise face verification.

Key defaults (in `config`)
- name: Yuri Xu | batch_size: 512 | lr: 0.0075 | epochs: 40 | num_classes: 8631 | seed: 5330

Dataset
Set `cls_data_dir` and `ver_data_dir` in the notebook. Optionally enable `get_data = True` to attempt a Kaggle download (requires credentials).

Transforms & Augmentations
`create_transforms(image_size=112, augment=True)` resizes to 112x112, converts to float tensor, applies optional random flip/rotation/resized-crop/color-jitter/blur, and normalizes with mean/std 0.5.
Batch-level CutMix/MixUp is supported via a `collate_fn`.

Model
The notebook provides a custom `Resnet34` (outputs `{"feats","out"}`) and a `ResNet34Scratch` built from torchvision (no pretrained weights).

Training & Evaluation
- Embedding training: TripletMarginLoss with `TripletImageDataset` / `train_epoch_triplet`.
- Classification training/validation: CrossEntropy (validation), `train_epoch` supports MixUp/CutMix and uses mixed precision.
- Verification: L2-normalize embeddings and cosine similarity; metrics computed include EER, AUC, ACC, and TPR@FPR.

Checkpointing & Submission
Use `save_model`/`load_model` and set `checkpoint_dir`. The notebook can generate `model_metadata_*.json` and assemble a final submission zip via `create_submission_zip` (requires ACKNOWLEDGED=True and path variables set).

Important vars before submission
Set `ACKNOWLEDGED = True`, and supply `KAGGLE_USERNAME`/`KAGGLE_API_KEY` and `MODEL_METADATA_JSON`/`NOTEBOOK_PATH` when creating the final zip.

Quick start
1) Open `main.ipynb`. 2) Edit `config` paths/hyperparams. 3) Set `ACKNOWLEDGED = True`. 4) Run training/validation; run test cells to create `verification_submission.csv` for Kaggle.

Example (zsh):
KAGGLE_COMPETITION=11785-hw-2-p-2-face-verification-fall-2025
kaggle competitions download -c "$KAGGLE_COMPETITION" -p ./data && unzip -qo ./data/${KAGGLE_COMPETITION}.zip -d ./data

Last updated: extracted from `main.ipynb`
- tqdm
