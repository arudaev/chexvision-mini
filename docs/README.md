# From-Scratch Implementation — Writeup Stub

Text scaffold for the report section / standalone docs. Figures are specified
separately in [`figures/figures-mini.html`](figures/figures-mini.html) and will
be produced by the design tooling.

## 1. Motivation
The course grades *implementing neural networks from scratch* to demonstrate the
underlying mathematics. CheXVision's two production models use PyTorch autograd,
so this companion implements the maths by hand in NumPy. It is a fundamentals
demonstration and does not produce the headline AUC numbers.

## 2. Architecture
A multilayer perceptron on 32×32 grayscale chest X-rays (1024 inputs),
`[Linear → ReLU → Dropout] × k → Linear(1)`, producing a single logit for the
binary *normal vs abnormal* task. Weights use He initialisation.

## 3. Forward pass
- `Linear`: `y = xW + b`
- `ReLU`: `max(0, x)`
- Output logit → `BCEWithLogitsLoss` (fused sigmoid + BCE, log-sum-exp stable).

## 4. Backpropagation (hand-derived)
- `Linear`: `dW = xᵀ·dout`, `db = Σ dout`, `dx = dout·Wᵀ`
- `ReLU`: gradient passes only where the input was positive
- Loss: `dL/dz = (σ(z) − y) / N`
- The `Sequential` container threads `dout` backward through every layer.

## 5. Gradient checking (correctness proof)
Central differences `(L(w+ε) − L(w−ε)) / 2ε` match the analytic gradients to a
relative error below 1e-6 — see `chexvision_mini/gradcheck.py` and the test.

## 6. Optimization & regularisation
Hand-coded SGD (+momentum) and Adam; L2 penalty and inverted dropout.

## 7. Results
Populate from `artifacts/history.json` after a Kaggle run (train/val loss curve,
val accuracy, val AUC). Compare qualitatively, not numerically, with the PyTorch
models.

## 8. Terminology (important for the report)
- **Custom CNN (random init)** — CheXVision Model 1 (PyTorch, trained from scratch *weights*).
- **Transfer DenseNet** — CheXVision Model 2 (PyTorch, pretrained DenseNet-121).
- **From-scratch NumPy net** — *this* repo (no framework, no autograd).
