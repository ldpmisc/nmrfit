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
```

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
Interested users are encouraged to test with saturation-recovery or Hanh echo data.
Example data will be uploaded in the future.

Currently only support .json input file but not raw data from NMR softwares like Bruker or MNova.
Current work-around is to load raw data using sSnake (https://doi.org/10.1016/j.jmr.2019.02.006, https://gitlab.science.ru.nl/mrrc/nmrzoo/ssnake) and export to a .json file.
The program will automaticlly detect 1D or 2D data. 

Click add peak and follow the instructions at the bottom left corner to add a peak. A peak will appear on the spectrum and peak table.
Right click on a peak table to add constraint. A constraint is a mathematical relations between two parameters used for a fitting. Addition of a constraint help reducing the number of independent fit variables. The dependent variable is called target, where the independent one is called driver, e.g driver --> target. Once clicked, a linkEditDialog shows up. Currently, the dialog only support 2 types of constraints, either Linear or RelaxDecayConstraint. Use the Dialog to set desired constraint.

Another way to set constraint is to use LinkManagerDialog by click on Link Manager buttons. This supports an additional RelaxGrowthConstraint type by directly typing on the Table. The parameter names are follows the synmatic sXX_p_YY_name where s stands for s, p stands for peak, XX and YY are corresponding number, name can be amp (amplitude), pos (position), gauss (gaussian width), and lor (lorentizian width). An example is s0_p1_amp (amplitude of peak 1 in slice 0). The dialog also supports import/export and mass generation of constraints. Follow the examples, which located at the bottom of the Table, to create proper constraints. 

Before fitting, it is encouraged to click validate button below the peak table to ensure that all constraints make sense (e.g no cyclic dependents that means a parameter A depends on paramter B but at the same time B depends on A). Note that each target can have only one driver and one relation. That means a target can not be controlled by 2 drivers, e.g A = B + C. However, one driver can be linked to multiple target, e.g A --> B and A --> C. A sequential relation such as A --> B --> C is also supported.

It is also possible to exclude a part of spectrum using "excluded" button.

To fit, click on fit button and select either "current slice", or "Sequential", or "Joint mode".

After fitting, it is possible to examine certain statistic. This feature is under development.

Click export to save the fit result in .txt file. 


