# funPVLBM

Main algorithm for *Penalized latent block model for matrix-valued functional data*.
The implementation combines Gaussian spline block models, penalty-guided class
reduction, fixed-order variational refinement and ICL-BIC selection. The optional
merging step adds independent row- and column-merge candidates.

This repository provides the main fitting program, numerical checks and one
small example. It does not contain the large Monte Carlo studies or their
per-repetition archives.

## Download and install

Download the repository using **Code → Download ZIP**, then extract it. Alternatively:

```sh
git clone https://github.com/liuy0135/funPVLBM.git
cd funPVLBM
```

Using Python 3.14, run the following from the extracted repository folder on macOS or Linux:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_installation.py
python example.py
```

The recorded dependency versions were tested with Python 3.14.7. The installation
check compares the working likelihood and block updates against direct numerical
calculations. The example runs one 30-by-30 functional matrix and saves its input,
selected parameters, memberships, labels and candidate summaries in `example_output/`.

## Fit your own data

```python
import numpy as np
from funpvlbm import fit

# y has shape (number of rows, number of columns, number of observation times).
# Replace unavailable measurements with np.nan; do not impute them for fitting.
data = np.load("my_data.npz")
result = fit(data["y"], data["times"], seed=19)
print(result["selected"])
np.savez_compressed("selected_fit.npz", **result["state"],
                    row_labels=result["row_labels"],
                    column_labels=result["column_labels"])
```

Observation times are scaled to `[0,1]`. `times` may be a common one-dimensional
grid or an array with the same shape as `y` for curve-specific times. Fitting uses
the observed values and their actual times. The spline basis has seven cubic
B-splines with interior knots at 0.25, 0.5 and 0.75.

The default initial cluster counts are `(7,7)` and `(10,10)`, and each penalty is
selected from `{0.02,0.05,0.09}`. Specify `initial_counts` and `penalties` to change
these candidates. Initial cluster counts cannot exceed the corresponding data
dimensions. Penalty pairs that violate the truncation conditions are skipped.
Set `with_merging=False` for Basic. Both versions refine candidates with fixed
class numbers and select by the unpenalized complete-data ICL-BIC score.

`result["state"]` contains `phi` and `gamma` (membership probabilities), `pi` and
`rho` (mixing proportions), `beta` (block spline coefficients) and `variance`.
Returned hard labels are zero-based. `result["records"]` includes convergence,
iteration counts, scores and recorded fitting failures for all attempted stages.

## Access the Lyon data

The original `velov` dataset is distributed in the R package
[funFEM](https://cran.r-project.org/package=funFEM), version 1.2, by Charles Bouveyron
(GPL-2). The dataset is obtained from CRAN rather than bundled here.

With R installed, run:

```sh
Rscript export_velov.R
```

This downloads the package's source data, removes the incomplete first day and
exports the complete Monday-to-Sunday week as `velov_week.csv`. To read it in Python:

```python
import numpy as np
from funpvlbm import fit
y = np.loadtxt("velov_week.csv", delimiter=",", skiprows=1).reshape(346, 7, 24)
times = np.linspace(0, 1, 24)
result = fit(y, times, initial_counts=((10, 7), (20, 7)), seed=19)
```

This call illustrates data access and fitting. The expanded search used for the
reported application is specified in the manuscript and Supplementary Section S1.2.

## Files

- `funpvlbm/model.py`: observed-data likelihood, block updates and penalized refinement.
- `funpvlbm/base_model.py`: initialization, mixing-proportion updates and merging.
- `funpvlbm/fit.py`: candidate generation and ICL-BIC selection.
- `example.py`: a small runnable example with automatically saved outputs.
- `check_installation.py`: numerical checks of the likelihood and updates.
- `export_velov.R`: access to the public Lyon data.
- `requirements.txt`: recorded Python dependency versions.

`VERIFICATION.json` records checks of this release, including agreement of its
fitting entry point with a saved, independently verified fit.
