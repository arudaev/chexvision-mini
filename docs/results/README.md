# Results data (for figures / design space)

Snapshot of the trained model's outputs, pulled from the HF model repo
(`arudaev/chexvision-mini`). Use these to build the results panel of Fig 2 and
any ROC/training-curve visuals — no need to re-run the model.

> Source: the **v5** Kaggle run (64×64 MLP). A follow-up **v6** run adds
> stream-shuffling to fix an unrepresentative test slice, so the *test* numbers
> below may shift in the final version; the validation numbers and the overall
> story are stable.

## Headline numbers (this snapshot)

| Metric | Test | Validation |
|---|---|---|
| ROC-AUC | 0.606 | **0.699** |
| Accuracy | 0.637 | 0.651 |
| F1 | 0.741 | 0.609 |

- Operating threshold (Youden's J on validation): 0.376.
- Class balance — train 40% / val 42% / **test 64%** positive. The test split
  (official NIH `test`, unshuffled first-10k slice here) is positive-heavy, which
  is why **ROC-AUC and balanced accuracy are the trustworthy metrics**, not plain
  accuracy/F1. (v6's shuffle addresses the slice representativeness.)

## Files

- `metrics.json` — full bundle: per-split scalar metrics, **downsampled ROC & PR
  curves** (200 pts), confusion matrices, and the run config.
- `history.json` — per-epoch `train_loss`, `reg_loss`, `val_loss`, `val_acc`,
  `val_auc`, `lr` (200 epochs) → the training-curve figure.
- `loss_curve.png` — the auto-generated train/val loss + val-AUC plot (reference).
- `val_scores.npy` / `val_labels.npy`, `test_scores.npy` / `test_labels.npy` —
  raw probabilities + 0/1 labels, to rebuild any ROC/PR/threshold figure at full
  resolution.
