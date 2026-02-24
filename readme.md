# nmrFit

**Physically-Constrained Multi-Slice Parametric Fitting for NMR Spectra**

nmrFit is a Python framework for quantitative fitting of multi-slice NMR datasets using physically meaningful parameter constraints and statistically diagnostics.

The software is designed for experiments where parameters evolve across slices (e.g., relaxation delay, temperature, field strength, composition) and independent fitting leads to unstable or highly correlated solutions.

---

# 1. Scientific Motivation

Conventional NMR spectral fitting often treats each spectrum independently:

* Parameters are optimized per slice
* Cross-slice physical relationships are ignored
* Correlations inflate uncertainties
* Identifiability degrades in overlapped systems

nmrFit enables joint optimization across multiple slices while enforcing explicit parameter relationships such as:

* Linear linking
* Exponential decay / growth
* Relaxation time parameterization (T ↔ k)
* Custom symbolic constraints


# 2. Mathematical Framework

Currently only support Gaussian-Lorentz peak shape

# 3. Fitting Modes

nmrFit supports three fitting strategies:

### Single Mode

Each slice fitted independently.

### Sequential Mode

Slices fitted in order with parameter propagation.

### Joint Mode

Simultaneous optimization across selected slices with shared or linked parameters.

Joint mode improves:

* Parameter stability
* Confidence interval reliability
* Physical interpretability

---

# 4. Statistical Diagnostics

For varied independent parameters, the framework provides:

* Covariance matrix
* Correlation matrix
* Strong correlation extraction (|r| thresholding)
* χ² and reduced χ²
* AIC and BIC model comparison

These diagnostics are intended to evaluate:

* Parameter identifiability
* Overfitting risk
* Model degeneracy
* Cross-slice coupling strength

---

# 5. Architecture Overview

The software is structured to separate UI, constraint logic, and fitting backend.

High-level flow:

```
FitContext
   ↓
ConstraintRule
   ↓
ConstraintStore
   ↓
FitOrchestrator
   ↓
lmfit Minimizer
   ↓
Statistical Extraction Layer
```

Design principles:

* Explicit parameter naming (e.g., s{slice}*p{peak}*{param})
* Deterministic constraint parsing
* No implicit parameter mutation
* Clear separation of model definition and optimization
* Extensible rule system

---

# 6. Core Features

* Multi-slice joint fitting
* Rule-based cross-slice parameter linking
* Relaxation decay / growth constraints
* Covariance and correlation extraction
* Correlation heatmap visualization
* Exportable and importable constraint definitions
* GUI-based constraint management (PyQt)

---

# 7. Installation

Python ≥ 3.10 recommended.
1. Clone the repository

```bash
git clone https://github.com/ldpmisc/nmrfit.git
cd nmrFit

2. Create environment

```
conda create -n nmrfit python=3.11
conda activate nmrfit
```

Install dependencies:

```
pip install -r requirements.txt
```

Primary dependencies:

* numpy
* scipy
* lmfit
* matplotlib
* PyQt5

---

# 8. Intended Use Cases

* Relaxation series analysis
* Temperature-dependent NMR fitting
* Multi-field spectral comparison
* Parameter-coupled spectral modeling
* Quantitative deconvolution of overlapped resonances

Particularly suitable for systems where:

* Phase heterogeneity exists
* Independent fits produce unstable parameters
* Physical evolution across slices is known or hypothesized

---

# 9. Limitations

* Requires reasonable initial parameter guesses
* Assumes user-defined peak model
* Does not implement automatic model selection
* Large multidimensional datasets may require optimization tuning
* Constraint system assumes deterministic mapping (no Bayesian framework)

---

# 10. Development Status

Active research development.

Planned directions:

* Extend nonlinear linking rules
* Enhance multi-tab GUI
* Improve statistic analysis

API may evolve.

# 11. Example
Interested users are encouraged to test with saturation-recovery data.
Example data will be uploaded in the future.


