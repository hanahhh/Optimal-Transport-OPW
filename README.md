# For Developers

## Setup

### Install dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements/dev.txt
pre-commit install
```

### Download data set

Example:

```bash
python -m scripts.download_data --data wei
```

# Run experiments

- Step localization experiments

Example:

```bash
python -m src.experiments.step_localization.evaluate --algorithm=POW --keep_percentile 0.3 --reg 3 --use_unlabeled
```

Read src/evaluate.py for more details.

- Weizmann classification 1-nn experiments

Example :

```bash
python -m src.experiments.weizmann.knn_eval --test_size 0.5 --outlier_ratio 0.1 --metric pow  --m 0.9 --reg 1 --distance euclidean
```

Read src/experiments/weizmann/knn_eval.py for more details.
