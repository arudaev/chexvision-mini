# Full Book Read-Through — 2026-06-14

Range verification: ran `pdftotext -layout docs/references/ref1_NNSP.pdf /tmp/nnsp.txt` and derived chapter ranges from the PDF's own chapter title pages and repeated chapter headers. No standalone `Contents` / `Table of Contents` heading was found in the extracted text, so the chapter title pages are used as the authoritative TOC surrogate. PDF metadata reports 658 pages; `/tmp/nnsp.txt` contains 658 non-empty extracted pages.

## Ch 1 — Introducing Neural Networks (pp. 7–20)

* Read: pp 7–20 via pdftotext; figures rendered: 11
* Proof-of-read: "`output = activation(output)`"
* Core concepts:
  * Supervised learning uses features/samples and labels/targets/ground truths to fit mappings from inputs to desired outputs.
  * Dense layers connect each neuron to all outputs from the previous layer; weights scale inputs and biases shift outputs.
  * Weights affect slope/magnitude/sign while biases offset the function, and both are trainable parameters.
  * Inputs usually need numeric preprocessing such as scaling/normalization, and output layers may be multi-class or a single binary neuron.
  * Generalization requires training on in-sample data and checking out-of-sample validation data to detect memorization/overfitting.
* NNFS reference implementation: conceptual single-neuron / dense-layer forward formulas before classes are introduced.
* In chexvision-mini already?: partial — `chexvision_mini/layers.py:Linear` encodes dense weights/biases; `chexvision_mini/train.py:train` creates train/validation splits and standardizes; `chexvision_mini/data.py:binary_label` maps chest X-ray labels to a binary target.
* Gap / divergence vs book: the repo has the machinery but lacks a tiny visible "single neuron → dense layer → binary output" teaching artifact that mirrors the chapter's visual progression.
* Recommendation: add `docs/nnfs_alignment.md` with a short Ch. 1 style explanation tying `Linear`, preprocessing, binary output, validation, loss, and overfitting to the current X-ray pipeline.
* Maps to syllabus bullet(s): introduction to deep learning and neural networks; vectors/matrices/tensors; single neuron and multiple layers; training/validation/model evaluation; overfitting.
* Type: doc-only
* Effort: S   Priority: P2

## Ch 2 — Coding Our First Neurons (pp. 21–54)

* Read: pp 21–54 via pdftotext; figures rendered: 52
* Proof-of-read: "`layer_outputs = np.dot(inputs, np.array(weights).T) + biases`"
* Core concepts:
  * A neuron computes an input-weight sum plus a bias; a layer repeats that operation for multiple neurons.
  * Python lists can show the mechanics, but NumPy dot products make the same math shorter and faster.
  * Tensors/arrays/vectors/matrices are treated practically as homologous numeric containers with explicit shapes.
  * Batches are matrices of samples; matrix product plus transposition produces all sample-by-neuron outputs.
  * Bias broadcasting adds one bias per neuron across every sample in the batch.
* NNFS reference implementation: the chapter's canonical dense-layer batch expression is `np.dot(inputs, np.array(weights).T) + biases`.
* In chexvision-mini already?: yes — `chexvision_mini/layers.py:Linear.forward` computes `x @ W + b` for a batch; `chexvision_mini/data.py:load_xray_subset` returns `x` as `(n, image_size**2)`; `chexvision_mini/train.py:_iterate_minibatches` feeds batches.
* Gap / divergence vs book: equivalent matrix orientation, but the repo stores weights as `(in_features, out_features)` and therefore uses `x @ W + b` instead of NNFS's row-per-neuron weights plus `.T`.
* Recommendation: document this orientation difference in `docs/nnfs_alignment.md`; no core code change.
* Maps to syllabus bullet(s): vectors, matrices, tensors, and linear transformations; single neuron and multiple layers; implementing a network with NumPy; batch processing.
* Type: doc-only
* Effort: S   Priority: P2

## Ch 3 — Adding Layers (pp. 55–67)

* Read: pp 55–67 via pdftotext; figures rendered: 58
* Proof-of-read: "`self.output = np.dot(inputs, self.weights) + self.biases`"
* Core concepts:
  * A network becomes deep when layers exist between input and output; hidden layer output becomes the next layer's input.
  * Each new layer's input width must equal the previous layer's output neuron count.
  * A dense layer class owns weight/bias arrays, initializes them, and exposes a forward pass.
  * Small random weights and zero biases are a first baseline; initialization is a tunable design choice.
  * Non-linear toy datasets are useful for learning mechanics before moving to real datasets.
* NNFS reference implementation: `Layer_Dense.__init__` and `Layer_Dense.forward`.
* In chexvision-mini already?: yes — `chexvision_mini/layers.py:Linear` owns params/grads and implements batch forward; `chexvision_mini/network.py:Sequential` chains multiple layers; `chexvision_mini/train.py:build_mlp` builds `[Linear -> ReLU -> Dropout] * k -> Linear`.
* Gap / divergence vs book: chexvision-mini uses He initialization rather than NNFS Ch. 3's `0.01 * randn`, which is a deliberate later-stage improvement for ReLU but should be explained for course traceability.
* Recommendation: document the initialization difference in `docs/nnfs_alignment.md` and cite `chexvision_mini/layers.py:Linear` as the book-aligned dense layer equivalent.
* Maps to syllabus bullet(s): single neuron and multiple layers; implementing a network with NumPy; hyperparameters; working from toy data toward real datasets.
* Type: doc-only
* Effort: S   Priority: P2

## Ch 4 — Activation Functions (pp. 68–106)

* Read: pp 68–106 via pdftotext; figures rendered: 90, 100
* Proof-of-read: "`self.output = np.maximum(0, inputs)`"
* Core concepts:
  * Hidden layers need nonlinear activation functions to model nonlinear relationships; purely linear hidden layers collapse to a linear function.
  * Step, Linear, Sigmoid, ReLU, and Softmax serve different roles: hidden-layer nonlinearity, regression output, binary probability, or multi-class distribution.
  * ReLU is simple and efficient: it clips negative values to zero and lets positive values pass through.
  * ReLU neuron pairs create local "areas of effect"; more neurons/layers can approximate richer nonlinear curves.
  * Softmax exponentiates logits and normalizes per sample using `axis=1` / `keepdims=True`; subtracting the row max prevents overflow without changing probabilities.
* NNFS reference implementation: `Activation_ReLU.forward` and `Activation_Softmax.forward`.
* In chexvision-mini already?: partial — `chexvision_mini/layers.py:ReLU` matches hidden-layer ReLU; `chexvision_mini/layers.py:Sigmoid` and `chexvision_mini/losses.py:sigmoid` cover sigmoid for binary probabilities; no `Softmax` activation exists.
* Gap / divergence vs book: missing Softmax and multi-class confidence distribution machinery. This is a syllabus/book coverage gap only; the main X-ray model should stay one logit plus sigmoid/BCE.
* Recommendation: add `Softmax` to `chexvision_mini/layers.py` with stable max subtraction and tests for row sums/stability; document that it is for syllabus demos, not the main binary X-ray head.
* Maps to syllabus bullet(s): activation functions (ReLU, Sigmoid, Softmax); NumPy implementation; loss/model evaluation setup for classification.
* Type: syllabus-coverage
* Effort: M   Priority: P1

## Ch 5 — Calculating Network Error with Loss (pp. 107–126)

* Read: pp 107–126 via pdftotext; figures rendered: 108
* Proof-of-read: "`y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)`"
* Core concepts:
  * Accuracy alone ignores confidence; loss rewards higher confidence on correct classes and penalizes misplaced confidence.
  * Categorical cross-entropy with one-hot targets simplifies to negative log of the correct class confidence.
  * Sparse targets and one-hot targets require different indexing/masking paths but produce the same sample-loss idea.
  * Predicted probabilities are clipped away from 0 and 1 to avoid infinite or negative loss artifacts.
  * Batch loss is the arithmetic mean of per-sample losses; accuracy is `argmax` predictions compared with target labels.
* NNFS reference implementation: `Loss.calculate` and `Loss_CategoricalCrossentropy.forward`.
* In chexvision-mini already?: partial — `chexvision_mini/losses.py:BCEWithLogitsLoss` implements the binary Chapter 16-style loss, not categorical cross-entropy; `chexvision_mini/metrics.py:accuracy` handles binary threshold accuracy, and `evaluation_report` adds AUC/F1/confusion matrix.
* Gap / divergence vs book: missing categorical cross-entropy and sparse/one-hot multi-class loss path, which pairs with the missing Softmax from Ch. 4. This is not needed for the binary X-ray score path.
* Recommendation: add `CategoricalCrossEntropyLoss` or a combined `SoftmaxCrossEntropyLoss` in `chexvision_mini/losses.py` with tests, clearly labeled as syllabus/demo coverage only.
* Maps to syllabus bullet(s): loss functions and error computation; model evaluation and accuracy; Softmax/categorical classification foundations.
* Type: syllabus-coverage
* Effort: M   Priority: P1

## Ch 6 — Introducing Optimization (pp. 127–134)

* Read: pp 127–134 via pdftotext; figures rendered: none
* Proof-of-read: "`lowest_loss = 9999999`"
* Core concepts:
  * Optimization means changing weights and biases to reduce loss.
  * Pure random parameter search is ineffective even on simple data.
  * Small random perturbations with rollback can improve a simple vertical dataset but fail on harder spiral data.
  * Copying parameter arrays matters because retaining references would mutate the saved "best" values.
  * The failure mode motivates gradients and systematic optimizers rather than blind search.
* NNFS reference implementation: random-search and random-walk loops around `Layer_Dense`, ReLU, Softmax, and categorical cross-entropy.
* In chexvision-mini already?: yes for real optimization, no for the intentionally bad baseline — `chexvision_mini/optim.py:SGD` and `chexvision_mini/optim.py:Adam` already implement systematic optimizers; `chexvision_mini/train.py:train` saves best state by validation AUC.
* Gap / divergence vs book: no random-search teaching demo, but this is not needed for model quality.
* Recommendation: optional doc-only note in `docs/nnfs_alignment.md` explaining that chexvision-mini skips random search and starts at gradient-based optimizers because Ch. 6 exists only to motivate them.
* Maps to syllabus bullet(s): optimization; hyperparameters; training pipeline; loss/error minimization.
* Type: doc-only
* Effort: S   Priority: P2

## Ch 7 — Derivatives (pp. 135–161)

* Read: pp 135–161 via pdftotext; figures rendered: 149
* Proof-of-read: "`p2_delta = 0.0001`"
* Core concepts:
  * Random search is infeasible because neural networks have many parameters and each parameter affects loss differently.
  * A derivative is the instantaneous slope/rate of change of a single-variable function at a point.
  * Numerical differentiation approximates the tangent slope with two nearby points and a small delta.
  * Numerical differentiation is too expensive for neural-network training because it requires repeated forward passes for every parameter and sample.
  * Analytical derivatives use exact rules such as constant, linear, constant-multiple, sum/subtraction, and exponent rules.
* NNFS reference implementation: numerical derivative demonstrations around `approximate_derivative = (y2-y1)/(x2-x1)` plus the analytical derivative rules used before backprop.
* In chexvision-mini already?: partial — `chexvision_mini/gradcheck.py:gradient_check` uses finite differences as a correctness proof, while `chexvision_mini/layers.py`, `chexvision_mini/losses.py`, and `chexvision_mini/regularizers.py` use analytical backward methods for training.
* Gap / divergence vs book: the project has the right mechanism but does not explain the pedagogical transition from numerical differentiation to analytical backprop.
* Recommendation: add a short `docs/nnfs_alignment.md` section explaining that `gradient_check.py` mirrors Ch. 7 numerical differentiation for verification only, while training uses analytical gradients for speed.
* Maps to syllabus bullet(s): derivatives, gradients, and chain rule; backpropagation preparation; implementing neural networks from scratch.
* Type: doc-only
* Effort: S   Priority: P2

## Ch 8 — Gradients, Partial Derivatives, and the Chain Rule (pp. 162–175)

* Read: pp 162–175 via pdftotext; figures rendered: 171
* Proof-of-read: "The gradient is a vector of all possible partial derivatives."
* Core concepts:
  * A partial derivative measures the impact of one input on a multivariate function while treating other inputs as constants.
  * A gradient collects all partial derivatives into a vector shaped like the function's inputs.
  * The partial derivative of multiplication with respect to one input equals the other input, which is the core pattern behind dense-layer gradients.
  * The derivative of `max(x, 0)` is 1 for positive inputs and 0 otherwise, which is the ReLU backward mask.
  * The chain rule says chained-function derivatives multiply through the forward-pass graph from loss back to parameters.
* NNFS reference implementation: the chapter's expanded loss-as-a-chain diagram and derivative rules that feed into `Layer_Dense.backward` and activation backward classes in the next chapter.
* In chexvision-mini already?: yes — `chexvision_mini/layers.py:Linear.backward` implements dense-layer partials for inputs, weights, and biases; `chexvision_mini/layers.py:ReLU.backward` implements the `x > 0` mask; `chexvision_mini/losses.py:BCEWithLogitsLoss.backward` starts the chain from loss to logits; `chexvision_mini/network.py:Sequential.backward` applies the chain in reverse.
* Gap / divergence vs book: the code is aligned, but the repo does not yet include a readable derivation map from Ch. 8 rules to the exact backward methods.
* Recommendation: add a compact derivation table in `docs/nnfs_alignment.md` mapping `sum`, `multiply`, `max`, and chained functions to `Linear.backward`, `ReLU.backward`, `BCEWithLogitsLoss.backward`, and `Sequential.backward`.
* Maps to syllabus bullet(s): gradients, derivatives, and chain rule; backpropagation; matrix operations and vector algebra; implementing a network with NumPy.
* Type: doc-only
* Effort: S   Priority: P2

## Ch 9 — Backpropagation (pp. 176–244)

* Read: pp 176–244 via pdftotext; figures rendered: 210, 224
* Proof-of-read: "`self.dweights = np.dot(self.inputs.T, dvalues)`"
* Core concepts:
  * Backpropagation applies the chain rule from the loss backward through activation functions and dense layers.
  * Dense-layer gradients are `dweights = inputs.T @ dvalues`, `dbiases = sum(dvalues, axis=0, keepdims=True)`, and `dinputs = dvalues @ weights.T`.
  * ReLU backward copies incoming gradients and zeros entries whose stored forward inputs were `<= 0`.
  * Loss gradients should be normalized by the batch size so learning-rate behavior is less dependent on batch size.
  * Softmax backward is a Jacobian-vector product, but Softmax plus categorical cross-entropy simplifies to `(probabilities - one_hot_targets) / samples`.
* NNFS reference implementation: `Layer_Dense.backward`, `Activation_ReLU.backward`, `Activation_Softmax.backward`, `Loss_CategoricalCrossentropy.backward`, and `Activation_Softmax_Loss_CategoricalCrossentropy.backward`.
* In chexvision-mini already?: partial — `chexvision_mini/layers.py:Linear.backward` and `ReLU.backward` match the dense/ReLU formulas; `chexvision_mini/network.py:Sequential.backward` reverses the layer chain; `chexvision_mini/losses.py:BCEWithLogitsLoss.backward` normalizes `(sigmoid(logits) - targets) / N`, which is the binary analogue of the book's combined Softmax+CCE gradient.
* Gap / divergence vs book: the main binary model is aligned, but the repo does not include the book's multi-class `Softmax`, `CategoricalCrossEntropyLoss`, or combined Softmax+CCE class for syllabus coverage and gradient-check tests.
* Recommendation: add `Softmax` and `SoftmaxCrossEntropyLoss`/`CategoricalCrossEntropyLoss` as demo-compatible modules with gradient-check coverage; keep `BCEWithLogitsLoss` as the production X-ray head because binary normal-vs-abnormal is the project task.
* Maps to syllabus bullet(s): backpropagation and training; gradients, derivatives, and chain rule; activation functions; loss functions; NumPy implementation.
* Type: syllabus-coverage
* Effort: M   Priority: P1

## Ch 10 — Optimizers (pp. 245–318)

* Read: pp 245–318 via pdftotext; figures rendered: 273, 297, 303
* Proof-of-read: "`cache = rho * cache + (1 - rho) * gradient ** 2`"
* Core concepts:
  * SGD updates parameters by subtracting learning-rate-scaled gradients; mini-batch SGD is the practical deep-learning default.
  * Learning rate controls whether training stagnates, converges, oscillates, or explodes; decay lowers the rate over time.
  * Momentum keeps a fraction of the previous update so SGD can move through shallow local minima and reduce bouncing.
  * AdaGrad and RMSProp adapt the update per parameter using accumulated or moving-average squared gradients.
  * Adam combines RMSProp-style squared-gradient cache with momentum and bias correction, and is a strong default but not always the best optimizer.
* NNFS reference implementation: `Optimizer_SGD`, `Optimizer_Adagrad`, `Optimizer_RMSprop`, and `Optimizer_Adam` with `pre_update_params`, `update_params`, and `post_update_params`.
* In chexvision-mini already?: partial — `chexvision_mini/optim.py:SGD` implements SGD plus heavy-ball momentum; `chexvision_mini/optim.py:Adam` implements bias-corrected first/second moments; `chexvision_mini/train.py:_cosine_lr` provides schedule-based decay outside the optimizer. There is no RMSProp or AdaGrad.
* Gap / divergence vs book: RMSProp is explicitly in the course outline and book but absent from `optim.py`; AdaGrad is book-covered but less central and the course outline does not name it.
* Recommendation: add `RMSProp` to `chexvision_mini/optim.py`, wire it through `config.yaml` / `train.py`, and add optimizer-step unit tests. Keep Adam as the default X-ray optimizer unless a CPU ablation shows RMSProp or SGD+momentum improves validation AUC. Do not add AdaGrad unless a small syllabus-demo optimizer matrix is desired.
* Maps to syllabus bullet(s): optimization methods (SGD, Adam, RMSProp); hyperparameters and learning rate; training pipeline; model evaluation.
* Type: syllabus-coverage
* Effort: M   Priority: P1

## Ch 11 — Testing with Out-of-Sample Data (pp. 319–325)

* Read: pp 319–325 via pdftotext; figures rendered: 320, 324
* Proof-of-read: "Overfitting is effectively just memorizing the data without any understanding of it."
* Core concepts:
  * High training accuracy can hide memorization; the model must be challenged with previously unseen data.
  * Train and test/out-of-sample data have distinct purposes: training updates parameters, testing estimates generalization after training.
  * Data leakage can make test performance look better than true generalization, especially when samples are highly related.
  * Diverging train and test/validation loss is an overfitting signal.
  * Smaller models, fewer epochs, regularization, dropout, and hyperparameter search are tools for reducing overfitting.
* NNFS reference implementation: separate `X_test, y_test = spiral_data(...)` followed by a forward-only validation pass and loss/accuracy calculation.
* In chexvision-mini already?: partial — `chexvision_mini/train.py:train` uses separate train/validation loads, tracks train/validation loss and validation AUC, and restores the best validation checkpoint; `chexvision_mini/metrics.py:evaluation_report` reports metrics on the validation set.
* Gap / divergence vs book: the final reported metrics currently reuse the validation set that selects the best checkpoint, so they are slightly optimistic compared with a true untouched test/out-of-sample evaluation.
* Recommendation: add a final held-out evaluation path in `chexvision_mini/train.py` and `config.yaml` (`n_test` or `eval_split`) that loads a distinct split/slice after checkpoint selection, writes `test_scores.npy`, `test_labels.npy`, and test metrics, and clearly labels validation as model-selection data and test as final generalization evidence.
* Maps to syllabus bullet(s): training, validation, and model evaluation; overfitting; working with real datasets; inference/evaluation.
* Type: inference/eval
* Effort: M   Priority: P1

## Ch 12 — Validation Data (pp. 326–329)

* Read: pp 326–329 via pdftotext; figures rendered: 328
* Proof-of-read: "Hyperparameter tuning using the test dataset is a mistake."
* Core concepts:
  * Test data should be used only for the final generalization check, not for choosing hyperparameters.
  * Validation data exists for hyperparameter selection and model comparison.
  * If data is scarce, temporarily split training data for validation, then retrain the selected configuration on all training data before testing.
  * k-fold cross-validation rotates validation folds so every chunk is validated once, but it is mainly useful when data is small.
  * Hyperparameter search should be iterative and selective rather than an exhaustive sweep unless training is extremely cheap.
* NNFS reference implementation: conceptual validation/test split and k-fold cross-validation diagrams, not a new code class.
* In chexvision-mini already?: partial — `chexvision_mini/train.py:train` uses validation AUC for checkpoint selection and logs validation metrics; there is no explicit ablation runner or untouched test metric separation.
* Gap / divergence vs book: validation is being used correctly for checkpoint selection, but the repo has no disciplined, recorded validation-only hyperparameter search path and no final test-only claim boundary.
* Recommendation: add a small CPU-friendly ablation script or documented run matrix that compares only a few validation-selected settings (`hidden_dims`, dropout, weight decay, optimizer, learning rate) and writes an `experiments.json`; reserve the final test/eval split from Ch. 11 for one final report pass. Skip k-fold for the real HF stream unless the dataset subset is tiny and fixed locally.
* Maps to syllabus bullet(s): hyperparameters; learning rate and training pipeline; training/validation/model evaluation; overfitting.
* Type: score-impacting
* Effort: M   Priority: P1

## Ch 13 — Training Dataset (pp. 330–332)

* Read: pp 330–332 via pdftotext; figures rendered: none
* Proof-of-read: "`divide the whole dataset by 255`"
* Core concepts:
  * Preprocessing applied to training data must also be applied to validation, testing, and prediction data.
  * Neural networks usually train better when numeric inputs are scaled around `[0, 1]` or preferably around `[-1, 1]`.
  * Scaler parameters should be prepared from training data and reused for validation/test/inference to avoid leakage.
  * Image augmentation can improve generalization only when the transformations are plausible for the real task.
  * Required sample count depends on task complexity, number of classes, feature count, and model size.
* NNFS reference implementation: preprocessing guidance rather than a new class; image examples use division by 255 and optional centering.
* In chexvision-mini already?: yes — `chexvision_mini/data.py:_streamed_subset` scales pixels to `[0, 1]`; `chexvision_mini/train.py:train` computes train-only mean/std, applies them to train/validation, and saves `_norm_mean` / `_norm_std`; `chexvision_mini/augment.py:augment_batch` applies simple image augmentations.
* Gap / divergence vs book: project docs disagree on 32x32 vs 64x64, and the augmentation choices are not explicitly justified as label-preserving for binary chest X-ray classification.
* Recommendation: fix the input-size documentation mismatch (`the development guide`, `docs/README.md`, root README/model card text) to match `config.yaml:image_size`, and document the preprocessing contract: pixel scaling, train-only standardization, saved normalization stats, and why/when horizontal flip/noise/brightness are enabled.
* Maps to syllabus bullet(s): working with real datasets; data pipelines and preprocessing; training/validation/model evaluation; regularization/generalization through augmentation.
* Type: doc-only
* Effort: S   Priority: P1

## Ch 14 — L1 and L2 Regularization (pp. 333–358)

* Read: pp 333–358 via pdftotext; figures rendered: 341, 355
* Proof-of-read: "`loss = data_loss + l1w + l1b + l2w + l2b`"
* Core concepts:
  * Regularization adds a penalty to data loss to reduce generalization error and discourage large parameters.
  * L1 uses absolute parameter values and has a piecewise gradient of `+1` / `-1`.
  * L2 uses squared parameter values and adds a gradient proportional to `2 * lambda * parameter`.
  * Regularization loss should be logged separately from data loss so the total loss is interpretable.
  * Regularization can reduce train/test divergence, and more data can reduce overfitting even more directly.
* NNFS reference implementation: dense-layer constructor regularizer strengths, `Loss.regularization_loss`, and `Layer_Dense.backward` additions for L1/L2 on weights and biases.
* In chexvision-mini already?: partial — `chexvision_mini/optim.py` applies `weight_decay * param` during SGD/Adam updates; `chexvision_mini/regularizers.py:l2_penalty` computes `0.5 * wd * sum(W**2)` over weights only; `chexvision_mini/train.py` does not log regularization loss; no L1 implementation exists.
* Gap / divergence vs book: regularization is split between optimizer weight decay and an unused reporting helper; biases are included in optimizer weight decay but excluded from `l2_penalty`; `regularizers.py` claims L1/L2 coverage while only L2 is present.
* Recommendation: make regularization internally consistent: add optional L1 penalty/gradient support or correct the docstring; decide explicitly whether L2 excludes biases, then make `SGD`/`Adam` and `l2_penalty` match; log `data_loss`, `reg_loss`, and `total_loss` in `history.json` / console output.
* Maps to syllabus bullet(s): regularization and overfitting; L1/L2; loss functions; backpropagation; model evaluation.
* Type: score-impacting
* Effort: M   Priority: P1

## Ch 15 — Dropout (pp. 359–385)

* Read: pp 359–385 via pdftotext; figures rendered: 365, 367, 383
* Proof-of-read: "`self.dinputs = dvalues * self.binary_mask`"
* Core concepts:
  * Dropout randomly zeroes neuron outputs during training so the model cannot depend too heavily on specific neurons.
  * The dropout rate can mean "disable probability" or "keep probability" depending on implementation, so semantics must be explicit.
  * Inverted dropout divides the mask by the keep probability during training, so inference can omit dropout without rescaling.
  * The backward pass multiplies incoming gradients by the same scaled mask used in the forward pass.
  * Dropout can reduce overfitting but may reduce training accuracy/loss and may require retuning model size and learning rate.
* NNFS reference implementation: `Layer_Dropout.forward` creates a scaled binomial mask and `Layer_Dropout.backward` multiplies gradients by that mask.
* In chexvision-mini already?: yes — `chexvision_mini/regularizers.py:Dropout` implements inverted dropout with `p` as disable probability, uses `training=False` as a no-op at inference, and reuses `_mask` in backward; tests cover train/eval behavior.
* Gap / divergence vs book: no major implementation gap. Minor guardrail: `Dropout.__init__` does not reject invalid `p >= 1` or `p < 0`; the current default dropout of `0.3` should be validated against smaller values because the book's worked examples often use `0.1` after other regularization.
* Recommendation: keep the dropout implementation, add `p` validation (`0 <= p < 1`), and include dropout rate in the validation ablation matrix (`0.0`, `0.1`, `0.2`, `0.3`) before deciding the final config.
* Maps to syllabus bullet(s): dropout; regularization and overfitting; training/validation/model evaluation; hyperparameters.
* Type: score-impacting
* Effort: S   Priority: P1

## Ch 16 — Binary Logistic Regression (pp. 386–420)

* Read: pp 386–420 via pdftotext; figures rendered: 388, 393, 416
* Proof-of-read: "`predictions = (activation2.output > 0.5) * 1`"
* Core concepts:
  * Binary logistic regression uses sigmoid output instead of softmax and binary cross-entropy instead of categorical cross-entropy.
  * A single output neuron can represent two classes, with labels reshaped as `(n_samples, 1)`.
  * Sigmoid maps logits to probabilities in `[0, 1]`, and its derivative is `sigmoid(x) * (1 - sigmoid(x))`.
  * Binary cross-entropy averages per-output losses and normalizes gradients by outputs and samples.
  * Binary classification accuracy can be computed by thresholding probabilities at `0.5`.
* NNFS reference implementation: `Activation_Sigmoid`, `Loss_BinaryCrossentropy`, one-output dense layer, sigmoid forward pass, BCE backward pass, and thresholded accuracy.
* In chexvision-mini already?: yes — `chexvision_mini/train.py:build_mlp` ends with `Linear(..., 1)`; `chexvision_mini/losses.py:BCEWithLogitsLoss` fuses sigmoid and BCE into a stable logit-space loss with gradient `(sigmoid(logits) - targets) / N`; `chexvision_mini/metrics.py:accuracy` thresholds probabilities at `0.5`; `chexvision_mini/data.py:binary_label` defines normal vs abnormal labels.
* Gap / divergence vs book: the implementation is more numerically stable than the book's separate sigmoid plus clipped BCE path, but the report/docs should explicitly explain this equivalence. Label smoothing is an extra practical regularizer not covered in this chapter.
* Recommendation: keep the one-logit BCE-with-logits design as the main X-ray model; document the equivalence to NNFS Ch. 16's sigmoid+BCE; include `label_smoothing` (`0.0` vs current `0.05`) in the validation ablation before claiming it improves results.
* Maps to syllabus bullet(s): Sigmoid activation; binary classification; loss functions and error computation; model evaluation; training with real datasets.
* Type: score-impacting
* Effort: S   Priority: P1

## Ch 17 — Regression (pp. 421–472)

* Read: pp 421–472 via pdftotext; figures rendered: 422, 447, 455
* Proof-of-read: "`self.output = inputs`"
* Core concepts:
  * Regression predicts scalar values rather than classes, so it uses a linear output activation instead of sigmoid/softmax.
  * MSE penalizes squared prediction error; MAE penalizes absolute error and is more robust but less commonly used.
  * Regression "accuracy" is approximate and depends on a task-specific tolerance/precision.
  * More depth can be necessary for fitting nonlinear scalar functions such as sine data.
  * Weight initialization can decide whether training learns or gets stuck; fan-aware initializers such as Glorot/He are preferable to a fixed magic scale.
* NNFS reference implementation: `Activation_Linear`, `Loss_MeanSquaredError`, `Loss_MeanAbsoluteError`, sine-data regression training loop, and an initialization comparison.
* In chexvision-mini already?: partial — the binary classifier already uses a linear final layer that emits raw logits before `BCEWithLogitsLoss`; `chexvision_mini/layers.py:Linear` uses He initialization, which directly addresses the chapter's initialization lesson. There are no MSE/MAE regression losses.
* Gap / divergence vs book: regression-specific losses and metrics are absent, but the course/project task is binary chest X-ray classification, not scalar regression.
* Recommendation: do not add regression losses to the main implementation path. Mention in `docs/nnfs_alignment.md` that Ch. 17 is intentionally skipped except for the transferable initialization lesson, already reflected by He initialization in `Linear`.
* Maps to syllabus bullet(s): activation functions; loss functions; hyperparameters/initialization; training pipeline.
* Type: skip
* Effort: S   Priority: P2

## Ch 18 — Model Object (pp. 473–530)

* Read: pp 473–530 via pdftotext; figures rendered: 478, 492, 528
* Proof-of-read: "`output = self.forward(X, training=True)`"
* Core concepts:
  * A model object wires layers, loss, optimizer, and accuracy into one reusable training/evaluation interface.
  * Forward propagation can be expressed as a chain of objects, where each layer reads the previous object's output.
  * Backward propagation reverses the same chain, beginning from the loss gradient and passing each object's `dinputs` to the previous layer.
  * Trainable layers should be tracked explicitly so optimizers update only parameterized layers.
  * Validation/inference forward passes must run with `training=False`, especially so dropout becomes a no-op.
* NNFS reference implementation: `Model.add`, `Model.set`, `Model.finalize`, `Model.forward`, `Model.backward`, and `Model.train`, including combined softmax-cross-entropy handling and printed `data_loss` / `reg_loss`.
* In chexvision-mini already?: partial — `chexvision_mini/network.py:Sequential` provides the layer chain, forward/backward pass, `params_and_grads`, and state dict; `chexvision_mini/train.py:train` owns loss/optimizer/metrics/checkpointing; `chexvision_mini/regularizers.py:Dropout.forward` already accepts `training=False`.
* Gap / divergence vs book: the project has the mechanics but not a small model-level inference/loading interface, and training logs do not expose the book's `data_loss` vs `reg_loss` split. A full NNFS-style `Model` class would be mostly a refactor, not a score win.
* Recommendation: do not replace `Sequential` with a large `Model` wrapper. Instead, add a thin `chexvision_mini/inference.py` or `network.load_checkpoint` helper later that reconstructs the configured MLP, loads `model.npz`, applies saved normalization stats, and exposes `predict_proba` / `predict_label`; also fold the Ch. 14 `data_loss`, `reg_loss`, `total_loss` logging into `train.py`.
* Maps to syllabus bullet(s): implementing a network with NumPy; backpropagation and training pipelines; dropout/inference behavior; model evaluation and inference.
* Type: inference/eval
* Effort: M   Priority: P1

## Ch 19 — A Real Dataset (pp. 531–592)

* Read: pp 531–592 via pdftotext; figures rendered: 535, 545, 548, 567
* Proof-of-read: "`X = X[keys]`"
* Core concepts:
  * Real image data must be loaded with labels, inspected visually, scaled consistently, flattened for dense layers, and split into train/test sets.
  * Class balance matters: a model can reduce loss by predicting the majority class if data are skewed or ordered by label.
  * Training data should be shuffled together with labels before batch training; otherwise batches can push the model toward whichever class appears in the current chunk.
  * Mini-batch training loops need step counts that include the final partial batch.
  * Epoch loss/accuracy should be accumulated by sample count, not only by per-batch averages.
* NNFS reference implementation: `load_mnist_dataset`, `create_data_mnist`, key-based shuffling, `Model.train(..., batch_size=...)`, and accumulated `Loss` / `Accuracy` statistics.
* In chexvision-mini already?: partial — `chexvision_mini/data.py:_streamed_subset` streams real HF images, converts to grayscale, resizes, scales to `[0, 1]`, flattens, and maps labels to binary; `chexvision_mini/train.py:_iterate_minibatches` shuffles loaded training samples each epoch with `rng.permutation`; `train.py` standardizes with train-only mean/std and uses mini-batches.
* Gap / divergence vs book: the loaded training subset is shuffled before batching, but the HF streaming subset itself is currently `stream.take(n_samples)` with no upstream shuffle or class-balance control. The console prints only train positive rate, not train/val/test class counts. Epoch train loss is `np.mean(epoch_losses)`, which gives the final partial batch the same weight as a full batch.
* Recommendation: in `chexvision_mini/data.py`, shuffle the HF stream deterministically before `take` when supported (`stream.shuffle(seed=seed, buffer_size=...)`), or document why the source order is already randomized; add class-count/positive-rate logging for every split and optionally a simple balanced sampling mode for binary normal/abnormal. In `chexvision_mini/train.py`, compute epoch train loss as a sample-weighted mean across batches and record `n_train_pos`, `n_train_neg`, `n_val_pos`, `n_val_neg` in `metrics.json`.
* Maps to syllabus bullet(s): working with real datasets; data pipelines and preprocessing; batch vs stream processing; scalable ML workflows; training/validation/model evaluation.
* Type: score-impacting
* Effort: M   Priority: P0

## Ch 20 — Model Evaluation (pp. 593–599)

* Read: pp 593–599 via pdftotext; figures rendered: 593, 596, 598
* Proof-of-read: "`model.evaluate(X_test, y_test)`"
* Core concepts:
  * Evaluation should be a reusable operation, separate from the training loop.
  * Evaluation forward passes run with `training=False`, so dropout and other training-only behavior are disabled.
  * Evaluation can be batched just like training/validation to avoid memory pressure.
  * Validation data is used during model selection; testing data is used after choosing the final model/hyperparameters.
  * Evaluating on the training data after training can differ from accumulated epoch training metrics because the model changed during that epoch.
* NNFS reference implementation: `Model.evaluate(X_val, y_val, batch_size=None)` extracted from the validation block in `Model.train`.
* In chexvision-mini already?: partial — `chexvision_mini/train.py:evaluate` computes forward-only validation accuracy/AUC with `training=False`, and `metrics.py:evaluation_report` builds a richer report; however, final metrics are still produced on the validation split used for checkpoint selection.
* Gap / divergence vs book: there is no standalone public evaluation entry point that loads a saved checkpoint and evaluates an arbitrary split/file; there is also no untouched final test pass after hyperparameter/model selection.
* Recommendation: add a reusable batched evaluation helper that returns BCE loss, accuracy, AUC, and report metrics for any split, and wire `train.py` to run it on validation during training plus an optional final test split after restoring the best checkpoint. Keep validation and test artifacts clearly named (`val_*` vs `test_*`).
* Maps to syllabus bullet(s): training, validation, and model evaluation; inference; data pipelines; working with real datasets.
* Type: inference/eval
* Effort: M   Priority: P1

## Ch 21 — Saving and Loading Models and Their Parameters (pp. 600–615)

* Read: pp 600–615 via pdftotext; figures rendered: 608, 611, 614
* Proof-of-read: "`model.save_parameters('fashion_mnist.parms')`"
* Core concepts:
  * Trainable parameters should be retrievable and settable so a trained model can be inspected, transferred, or restored.
  * Saving parameters alone is smaller and simpler, but the exact model architecture must be recreated before loading them.
  * Saving a full model preserves structure and optimizer state, which is useful for resuming training.
  * Runtime-only tensors and gradients should not be persisted unnecessarily.
  * Loading should produce a ready-to-evaluate or ready-to-predict model.
* NNFS reference implementation: `Layer_Dense.get_parameters`, `Layer_Dense.set_parameters`, `Model.get_parameters`, `Model.set_parameters`, `Model.save_parameters`, `Model.load_parameters`, `Model.save`, and static `Model.load`.
* In chexvision-mini already?: partial — `chexvision_mini/network.py:Sequential.state_dict` and `load_state_dict` cover parameter retrieval/restoration; `chexvision_mini/train.py` saves `model.npz` plus `_norm_mean` / `_norm_std`; `metrics.json` stores a partial config snapshot.
* Gap / divergence vs book: loading is not yet a first-class workflow. The saved parameter file does not by itself describe the architecture, preprocessing contract, threshold, label mapping, dataset revision, or optimizer state. The current artifact is good for final inference but incomplete for painless resume/training continuation.
* Recommendation: keep the transparent `.npz` parameter format instead of switching to pickle as the default. Add a companion `metadata.json` or expand `metrics.json` with architecture, input dimension, image size, label mapping, normalization stats reference, threshold, dataset revision, code/package version, and training mode; add a `load_checkpoint(output_dir)` helper that reconstructs `build_mlp(...)`, loads `model.npz`, validates shapes, and returns a ready inference/evaluation object. Saving optimizer state is optional and only worth it if the project needs resume training.
* Maps to syllabus bullet(s): training pipeline; model evaluation and inference; scalable workflows/artifact handoff; implementation with NumPy.
* Type: inference/eval
* Effort: M   Priority: P1

## Ch 22 — Prediction / Inference (pp. 616–658)

* Read: pp 616–658 via pdftotext; figures rendered: 616, 623, 628, 653
* Proof-of-read: "`return np.vstack(output)`"
* Core concepts:
  * Prediction is a separate forward-only workflow that may still need batching for memory safety.
  * Single samples must be shaped as a batch, e.g. `(1, n_features)`, before entering the network.
  * Batched prediction outputs should be collected efficiently and stacked once at the end.
  * Raw model outputs must be converted into user-facing labels or probabilities.
  * Inference preprocessing must match training preprocessing exactly; even simple image polarity/scale differences can break a dense image classifier.
* NNFS reference implementation: `Model.predict(X, batch_size=None)`, output activation `predictions(...)`, label-name mapping, and image preprocessing for new external samples.
* In chexvision-mini already?: partial — `train.py` saves normalization stats and can compute validation probabilities after training; `losses.py:sigmoid` converts logits to probabilities; `metrics.py` thresholds probabilities at `0.5`. There is no standalone prediction helper or CLI that loads an artifact and preprocesses a new X-ray image.
* Gap / divergence vs book: the repo can train and report validation metrics, but it does not yet demonstrate the complete book-style path from saved checkpoint + raw image to binary prediction. That makes preprocessing drift more likely and weakens the report's inference story.
* Recommendation: add a small `chexvision_mini/inference.py` with `preprocess_image(path, image_size, mean, std)`, `predict_proba(model, x, batch_size=None)`, and `predict_label(prob, threshold=0.5)`; expose it through `python -m chexvision_mini predict --checkpoint artifacts --image path.png`. The binary output should be explicit: probability of abnormal, thresholded label (`normal` / `abnormal`), and the threshold used. Reuse the same grayscale/resize/flatten/`/255` + saved standardization path as training.
* Maps to syllabus bullet(s): inference; training with real datasets; data pipelines; model evaluation; Deep Learning integration in larger workflows.
* Type: inference/eval
* Effort: M   Priority: P1

## Consolidated v5 Plan (from full read)

This supersedes the earlier tier list in this file. The book read makes one thing clear: chexvision-mini is already correctly shaped as a pure-NumPy, dense, binary classifier. The best next work is not to make it more advanced; it is to make the current basics stricter, more measurable, and easier to use.

Verification note: I read every chapter text range from the `pdftotext -layout` extraction and rendered representative visual/code-heavy pages for each figure-heavy chapter. The PDF did not expose a clean formal TOC in extracted text, so chapter page ranges were derived from chapter title/header pages. I did not render every one of the 658 pages; recommendations below rely on full extracted chapter text plus selected visual verification.

### A) syllabus-coverage modules that do not alter X-ray scoring

* Add `Softmax` and categorical cross-entropy/combined softmax-cross-entropy as demo/test coverage only (`chexvision_mini/layers.py`, `chexvision_mini/losses.py`, `tests/`). Chapters 4, 5, and 9 require this syllabus concept, but the main X-ray task should stay one-logit BCE from Chapter 16.
* Add `RMSProp` to `chexvision_mini/optim.py` and config selection as syllabus coverage. Chapter 10 and the course list name RMSProp; keep Adam as default unless validation ablation proves otherwise.
* Add a short `docs/nnfs_alignment.md` mapping neuron → dense layer → activations → loss → chain rule/backprop → optimizers → regularization → inference to this repo's files. This covers Chapters 1–10 and 16 without adding complexity to runtime code.
* Fix documentation mismatch around image size: current code/config are 64×64, while older docs mention 32×32. Do not reduce the model to 32×32 just to match stale prose; either document 64×64 as the current choice or include image size in a controlled ablation.

### B) score-impacting changes

* Treat stream ordering and class balance as P0. In `chexvision_mini/data.py`, deterministically shuffle the HF stream before `take(...)` when possible; otherwise document source ordering. Log normal/abnormal counts for train/val/test. If counts are skewed, prefer a simple balanced subset sampler before adding loss weights.
* Make epoch train loss sample-weighted in `chexvision_mini/train.py`, not a plain mean of batch means. This is small but directly aligns with Chapter 19's accumulated-loss logic.
* Make regularization internally consistent. Decide whether L2 applies to weights only or weights+biases, then align `optim.py` weight decay and `regularizers.py:l2_penalty`; log `data_loss`, `reg_loss`, and `total_loss`. Add L1 only as optional/default-off syllabus support, not because the X-ray run needs it.
* Add `Dropout` parameter validation (`0 <= p < 1`) and tune dropout instead of assuming `0.3` is right. Validate `0.0`, `0.1`, `0.2`, `0.3`.
* Run a small CPU-friendly validation ablation matrix and save it as `experiments.json`: optimizer (`adam`, `sgd`, optional `rmsprop`), learning rate, dropout, weight decay, label smoothing (`0.0` vs `0.05`), hidden dims, and possibly image size. Reserve the final test split for one report pass.
* Keep the binary architecture: `Linear(..., 1)` + `BCEWithLogitsLoss`. This is both more appropriate for normal/abnormal and more numerically stable than the book's separate sigmoid+BCE code.

### C) inference/eval/report polish

* Add an untouched final evaluation split/path in `config.yaml` and `chexvision_mini/train.py`: write `test_scores.npy`, `test_labels.npy`, and test metrics only after selecting the best checkpoint by validation AUC.
* Add a reusable batched evaluation helper that returns BCE loss, accuracy, AUC, confusion metrics, ROC/PR inputs, and class counts for any split. Use it for validation during training and final test after training.
* Add checkpoint metadata: architecture, input dimension, image size, label mapping, dataset repo/revision, train/val/test counts, threshold, normalization stats reference, seed, mode, and package/code version. Keep `.npz` parameters; do not switch to pickle as the default artifact.
* Add `chexvision_mini/inference.py` plus CLI routing (`python -m chexvision_mini predict ...`) that loads a checkpoint, preprocesses a raw image exactly like training, applies saved normalization stats, returns abnormal probability, thresholded label, and threshold used.
* Pick any operational threshold on validation only, then report test metrics at that threshold. Keep ROC-AUC as the headline metric for model comparison because it is threshold-independent.
* Update the README/report text to be explicit: this is a fundamentals NumPy MLP, not the PyTorch CheXVision CNN/DenseNet result. The value is transparent math, gradient checks, CPU streaming, and honest evaluation.

### D) explicitly skip

* Do not add PyTorch, TensorFlow, JAX, autograd, GPU code, or a from-scratch CNN to this repo. That would violate the repo/course objective.
* Do not replace the binary X-ray head with softmax/categorical cross-entropy. Add softmax only as syllabus/demo coverage.
* Do not refactor `Sequential` into a large NNFS-style `Model` object unless future changes make the current split genuinely painful. A thin loader/evaluator/predictor is enough.
* Do not implement Spark/distributed compute inside chexvision-mini. For Big Data alignment, document the stream/batch pipeline and keep Kaggle/HF orchestration in the parent repo.
* Do not spend time on regression losses for the main path. Chapter 17 is intentionally skipped except for initialization lessons already reflected by He initialization.
* Do not add exhaustive random search or k-fold over the HF stream. Use a small recorded validation matrix.
* Do not add AdaGrad unless the instructor explicitly wants every optimizer from the book; RMSProp closes the course-listed gap.
* Do not pickle full model objects as the primary artifact. `.npz` + JSON metadata is clearer and safer for a small educational repo.

### Ordered v5 implementation list

1. `chexvision_mini/data.py`, `chexvision_mini/train.py`, `config.yaml`: deterministic stream shuffle before `take`, split class-count logging, optional balanced subset mode, and sample-weighted epoch loss. Ch. 19. Worth doing first because it can directly affect validation AUC and prevents misleading streamed subsets.
2. `config.yaml`, `chexvision_mini/train.py`, `chexvision_mini/metrics.py`: add final test split/evaluation artifacts and keep validation-only model selection. Ch. 11, 12, 20. Worth doing before tuning so the final report has a clean claim boundary.
3. `chexvision_mini/regularizers.py`, `chexvision_mini/optim.py`, `chexvision_mini/train.py`: regularization consistency plus `data_loss`/`reg_loss`/`total_loss` logging; add dropout `p` validation. Ch. 14, 15, 18. Worth doing because it improves both correctness and interpretability.
4. `scripts/` or `chexvision_mini/experiments.py`: add a tiny ablation runner/config matrix and write `experiments.json`. Ch. 12, 15, 16, 19. Worth doing because current hyperparameters are plausible but not evidenced.
5. `chexvision_mini/optim.py`, `train.py`, tests: add RMSProp as selectable optimizer. Ch. 10. Worth doing for lecture alignment; only use it by default if ablation wins.
6. `chexvision_mini/layers.py`, `chexvision_mini/losses.py`, `tests/`: add stable Softmax and categorical CE/combined Softmax+CCE gradient tests. Ch. 4, 5, 9. Worth doing for syllabus/book alignment; not part of X-ray scoring.
7. `chexvision_mini/network.py` or new `chexvision_mini/checkpoint.py`, plus `train.py`: add metadata and `load_checkpoint(output_dir)` that reconstructs the model and validates shapes. Ch. 21. Worth doing because saved weights without architecture/preprocessing are fragile.
8. `chexvision_mini/inference.py`, `chexvision_mini/__main__.py`: add `predict` CLI for raw image → abnormal probability/label. Ch. 22. Worth doing because inference is the final book chapter and proves preprocessing is reusable.
9. `docs/README.md`, root `README.md`, `the development guide`, model card template, new `docs/nnfs_alignment.md`: fix 32×32 vs 64×64, document BCE-with-logits equivalence, preprocessing, why softmax is demo-only, and the honest limitations of the NumPy MLP. Ch. 1–3, 7–9, 13, 16–17. Worth doing because the report will otherwise undersell the good parts and overstate the model's role.
10. Only after the above: consider optional L1 default-off support and optimizer-state resume. Ch. 14, 21. Not urgent because neither is required for current scoring or core alignment.
