"""One small synthetic example, with automatically saved output."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import adjusted_rand_score
from funpvlbm import fit

rng = np.random.default_rng(20261120)
times = np.linspace(0, 1, 30)
rows = rng.permutation(np.repeat(np.arange(3), 10))
columns = rng.permutation(np.repeat(np.arange(3), 10))
mean = (rows[:, None, None] + (columns[None, :, None]+1)*times
        + np.sin(2*np.pi*times))
y = mean + rng.normal(0, .6, (30, 30, 30))
y[rng.random(y.shape) < .06] = np.nan
result = fit(y, times, seed=19)
out = Path('example_output')
out.mkdir(exist_ok=True)
np.savez_compressed(out/'input.npz', y=y, times=times, true_rows=rows, true_columns=columns)
np.savez_compressed(out/'fit.npz', **result['state'],
                    row_labels=result['row_labels'], column_labels=result['column_labels'])
summary = dict(selected=result['selected'],
               row_ARI=float(adjusted_rand_score(rows, result['row_labels'])),
               column_ARI=float(adjusted_rand_score(columns, result['column_labels'])))
(out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
(out/'candidates.json').write_text(json.dumps(result['records'], indent=2)+'\n')
print(json.dumps(summary, indent=2))
print('Outputs saved to', out.resolve())
