# The cost of data constraints in turbofan RUL prediction

Code and results for the paper *The cost of data constraints in turbofan remaining useful life prediction: a factorial study of fleet size, sensor coverage and recording interval* (Rashed Ul Karim, manuscript in preparation, 2026).

The study asks how much prediction accuracy an operator loses when it has few engines that ran to failure, few sensors, or sensor readings logged only every few flights. It varies all three on the four NASA C-MAPSS turbofan data sets and fits three models that need no GPU and no tuning: ridge regression, a random forest and LightGBM. A mean-life reference that uses no sensors is included for comparison. In total 15,552 models were fitted.

Main findings, in median test RMSE of the boosted trees (cycles, targets capped at 125):

| | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| All data (full fleet, 14 sensors, every flight) | 11.3 | 10.1 | 9.6 | 11.2 |
| 10 training engines | 16.9 | 16.1 | 16.9 | 17.5 |
| 4 legacy sensors | 13.0 | 12.9 | 9.8 | 12.5 |
| Every 20th flight recorded | 18.4 | 17.3 | 17.6 | 19.8 |

Fleet size explains about half of the variation in error, the recording interval a sixth to a third, and sensor coverage under a tenth. Adding the three penalties predicts the combined error to within about one cycle at the median.

## What is in this repository

| Path | Contents |
|---|---|
| `code/run_all.py` | Reruns the whole study with one command (see below) |
| `code/run_experiment.py` | The experiment for one data set. Restartable: finished parts are skipped |
| `code/analyse.py` | Builds the summary tables and Figures 1 to 3 |
| `code/stats_tests.py` | Wilcoxon tests with Holm adjustment, sign tests, and the logging versus sensor retrofit comparison |
| `code/sensitivity.py` | Hyperparameter sensitivity check (paper Section 4.7) |
| `code/analyse_sensitivity.py` | Builds Table 5 from the sensitivity runs |
| `code/check.py` | Checks on the full-data case: uncapped scoring, five seeds, and a held-out half of the training fleet |
| `results/FD00x.csv` | One row per fitted model, for each data set |
| `results/results_tests.csv` | All paired test results |
| `results/tables.txt` | All summary tables used in the paper |
| `results/sensitivity/` | Results of the sensitivity check and its summary |
| `figures/` | Figures 1 to 3 of the paper |

The NASA data files are not included. Download them as described below.

## How to reproduce the results

1. Install Python 3.10 or later, then install the libraries:

   ```
   pip install -r requirements.txt
   ```

2. Download the C-MAPSS data ("Turbofan Engine Degradation Simulation") from the NASA Open Data Portal: https://data.nasa.gov/dataset/simulating-degradation-data-for-prognostic-algorithm-development. Unzip it and note the folder that holds `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` and the same files for FD002 to FD004.

3. From the top folder of this repository, run:

   ```
   python code/run_all.py path/to/CMAPSSData
   ```

   This takes about one hour on two CPU cores. If it stops, run the same command again and it continues where it left off. Add `--sensitivity` at the end to also rerun the check in Section 4.7 (about 40 minutes more).

4. The tables appear in `out/tables.txt`, the figures in `out/`, and the test results in `results_tests.csv`. They should match the files in `results/` and `figures/`.

Each random draw of training engines uses a fixed seed made from the data set name, the fleet size and the draw number, so the same engines are drawn on every machine. Small differences in the last decimal can still appear with other library versions. The versions used for the paper are listed in `requirements.txt`.

## Columns in `results/FD00x.csv`

| Column | Meaning |
|---|---|
| `fd` | Data set (FD001 to FD004) |
| `k` | Recording interval: every k-th flight is kept (1, 5, 10, 20) |
| `n` | Number of training engines |
| `rep` | Draw number for that fleet size |
| `sensors` | Sensor set: S14, S7, S4, S2, or `none` for the mean-life reference |
| `model` | `Ridge`, `RF`, `GBM` or `MeanLife` |
| `rmse`, `mae` | Test errors in cycles, targets capped at 125 |
| `score` | NASA asymmetric score (late predictions cost more) |
| `bias` | Mean error (positive means late) |
| `late` | Share of test engines predicted more than 10 cycles late |
| `fit_s` | Training time in seconds |
| `n_rows` | Number of training rows |

## Data source

Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. 2008 International Conference on Prognostics and Health Management. https://doi.org/10.1109/PHM.2008.4711414

## How to cite

If you use this code or these results, please cite the repository using the details in `CITATION.cff`. A citation for the paper will be added once it is published.

## Licence

The code is released under the MIT licence (see `LICENSE`). The result files and figures may be reused with credit.
