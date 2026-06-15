# Results data (for figures / design space)

Snapshot of the trained model's outputs, pulled from the HF model repo
(`arudaev/chexvision-mini`). Use these to build any results panel / ROC /
training-curve visuals — no need to re-run the model.

> Source: the **v6** Kaggle run (64×64 MLP, stream-shuffled splits). Final.

## Headline numbers

| Metric | Test | Validation |
|---|---|---|
| ROC-AUC | 0.650 | **0.699** |
| Balanced accuracy | 0.590 | 0.652 |
| Accuracy | 0.647 | 0.654 |
| F1 | 0.744 | 0.608 |

- Operating threshold (Youden's J on validation): 0.389. Best epoch 176/200.
- **Class balance — train 0.41 / val 0.42 / test 0.62 positive.** The HF stream is
  shuffled before sampling, so these are representative: the higher test rate is a
  **real property of NIH ChestX-ray14's official `test` split**, not a sampling
  artifact. Because of that base-rate shift, **ROC-AUC and balanced accuracy are
  the trustworthy metrics** — plain accuracy/F1 on test are inflated by the 62%
  positive base rate. Validation AUC 0.699 reflects the model's quality; test AUC
  0.650 reflects the genuinely harder/different official test distribution.

## Files

- `metrics.json` — full bundle: per-split scalar metrics (incl. balanced accuracy),
  **downsampled ROC & PR curves** (200 pts), confusion matrices, and the run config.
- `history.json` — per-epoch `train_loss`, `reg_loss`, `val_loss`, `val_acc`,
  `val_auc`, `lr` (200 epochs) → the training-curve figure.
- `loss_curve.png` — the auto-generated train/val loss + val-AUC plot (reference).
- `val_scores.npy` / `val_labels.npy`, `test_scores.npy` / `test_labels.npy` —
  raw probabilities + 0/1 labels, to rebuild any ROC/PR/threshold figure at full
  resolution.
