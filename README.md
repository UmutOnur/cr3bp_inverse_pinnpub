# Recovering the CR3BP mass ratio with a PINN

Code and results behind the progress report: 599 of the 1000 orbits so far.
Every method uses 1000 noisy position samples per orbit (σ = 10⁻³).

## Files

- `train.py`: the PINN
- `baselines.py`: the comparison methods
- `demo.py`: one orbit end to end
- `summary.py`: prints the results table
- `results/`: the two result CSVs and the script that produced them
- `crb.py`: the best accuracy any method can reach at this noise
- `cr3bp_dataset_final.npy`, `data_gen.py`, `verify_dataset.py`: the dataset, how it was made, a check

## Running

```
pip install -r requirements.txt
pip install --no-build-isolation "git+https://github.com/nbelakovski/cr3bp"
python demo.py          # orbit 646, about 3 min on a GPU (--quick to test the install)
python summary.py       # results table
```

## CSVs

`traj_index` orbit, `mu_true` true μ, `mu_pred` (PINN) or `mu_est` (other methods)
estimated μ, `error_pct` relative error. `group` is close (`min_r` < 0.05),
middle (0.05–0.5) or far (> 0.5), as in the report; `min_r` is the orbit's closest
approach to either mass. BLS used a single starting value on the 200 orbits where
`cost_first` is empty.
