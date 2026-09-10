# HAI case-study artifact

`hai_case_artifact_seed42.npz` contains the case-specific model-derived values
used in the representative seed-42 HAI case-study analysis:

- relative case time;
- prediction-error evidence;
- pressure-residual evidence;
- heat-exchange-residual evidence; and
- score responses before and after OAAD enhancement.

The artifact is read by `scripts/reproduce_hai_case_figure.py`. It does not
contain the raw HAI CSV files, a model checkpoint, non-public industrial data,
or per-sample attack/anomaly-label columns.
