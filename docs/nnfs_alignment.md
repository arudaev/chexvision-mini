# NNFS → chexvision-mini alignment

How this repo maps to *Neural Networks from Scratch in Python* (Kinsley &
Kukieła). The project follows the book's pure-NumPy, no-autograd approach; the
main task is **binary** chest-X-ray screening, so a single sigmoid/logit + BCE
replaces the book's softmax + categorical cross-entropy where they differ.

| NNFS chapter(s) | Concept | Where in this repo |
|---|---|---|
| 1–3 | Neuron → dense layer → deep network | `layers.py:Linear` (`x @ W + b`), `network.py:Sequential` |
| 4 | ReLU / Sigmoid activations | `layers.py:ReLU`, `layers.py:Sigmoid` |
| 5, 16 | Loss (BCE for binary) | `losses.py:BCEWithLogitsLoss` (fused sigmoid + BCE, log-sum-exp stable) |
| 6–9 | Derivatives, chain rule, backprop | `*.backward()` everywhere; `Sequential.backward` reverses the chain |
| 7 | Numerical differentiation | `gradcheck.py` (finite differences, *verification only*) |
| 10 | Optimizers (SGD, RMSProp, Adam) | `optim.py:SGD`, `optim.py:RMSProp`, `optim.py:Adam` |
| 11–12, 20 | Train / validation / test discipline | `train.py` (val selects checkpoint; untouched **test** split for final metrics) |
| 13 | Preprocessing (scale, train-only stats) | `data.py` (`/255`), `train.py` (train-only standardisation, saved in `model.npz`) |
| 14 | L1 / L2 regularisation | `optim.py` (gradients, weights-only), `regularizers.py:l1_penalty`/`l2_penalty` |
| 15 | Dropout (inverted) | `regularizers.py:Dropout` |
| 18 | Model object | `network.py:Sequential` + `train.py` (kept lightweight, not a full `Model` class) |
| 21 | Save / load parameters | `network.py:state_dict`/`load_state_dict`, `inference.py:load_checkpoint` |
| 22 | Prediction / inference | `inference.py` + `python -m chexvision_mini predict` |

## Deliberate divergences from the book

- **Binary head, not multi-class.** The X-ray task is normal vs abnormal, so the
  production model is one logit + `BCEWithLogitsLoss` (Ch. 16), more numerically
  stable than the book's separate sigmoid + clipped BCE.
- **He initialisation** (`layers.py:Linear`) instead of Ch. 3's `0.01 * randn` —
  the Ch. 17 lesson that initialisation matters, applied for ReLU.
- **Threshold is chosen, not assumed 0.5.** With ~42% prevalence the 0.5 default
  is skewed; the operating point is picked on validation (Youden's J) and applied
  to the test split (`metrics.py:best_threshold`, `train.py`).
- **64×64 grayscale** input (4096 features) — a resolution/compute trade-off for
  a CPU-only MLP, not from the book.

## Intentionally out of scope

- **Softmax / categorical cross-entropy** (Ch. 4/5/9): the syllabus covers them,
  but they are not in this repo because the task is binary. (Could be added as a
  demo-only module if full multi-class coverage is wanted.)
- **Regression losses** (Ch. 17): not applicable; only the initialisation lesson
  is used.
- **Convolutions, autograd, GPU, frameworks, Spark**: out of scope by design —
  the parent CheXVision repo is where PyTorch/CNN/cloud live.
