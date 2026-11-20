# Breast Cancer Histopathology Classification

A CSCI 4150 project at Rensselaer Polytechnic Institute. We use the
[BreaKHis dataset](https://www.kaggle.com/datasets/ambarish/breakhis/data)
to study how well a model distinguishes benign and malignant breast tissue
images. This is a course project, not a tool for clinical diagnosis.

## View evaluation results without training

1. Open `evaluation_results.html` in a browser.
2. Click **Choose File** and select `outputs/fold_metrics.csv` or a copy of
   the CSV shared by a teammate.
3. If you don't have a CSV, click **Load sample data** to try the page.
   Sample values are for demonstration and are not model results.

The page reads the CSV locally and shows malignant recall and specificity
for each fold. It highlights the fold with the lowest malignant recall.
No server, TensorFlow installation, or image dataset is needed to use the page.

For a terminal version, run:

```sh
python summarize_evaluation.py
```

The script defaults to `outputs/fold_metrics.csv`. To use a CSV stored
somewhere else, pass its path as an argument:

```sh
python summarize_evaluation.py path/to/fold_metrics.csv
```

## Generate evaluation results

From the repository root, check that the dataset PNGs are under
`BreaKHis_v1/histology_slides/breast`, or update `DATA_DIR` in
`efficient_b_0.py`. With TensorFlow, NumPy, pandas, and scikit-learn installed
in your project environment, run:

```sh
python efficient_b_0.py
```

This starts a full five-fold training and evaluation run. It creates
`outputs/` locally; you don't need an existing copy. Training can take time,
and the image cache and saved models can take substantial disk space.

| Output | Contents |
| --- | --- |
| `outputs/fold_metrics.csv` | Per-fold metrics used by the results page and summary script. |
| `outputs/oof_predictions.csv` | Image-level labels, fold IDs, and predicted malignancy scores. |
| `outputs/patient_predictions.csv` | Scores aggregated across images for each patient record in a fold. |
| `outputs/threshold.json` | Selected cutoff, target sensitivity, and aggregation setting. |
| `outputs/fold*_best.keras` | Saved model for each fold. |

The default target is 95% patient-level malignant recall. The script chooses
the highest cutoff that meets this target on the pooled out-of-fold predictions.
Because the same predictions are used to choose and score the cutoff, the
reported metrics at that cutoff are not independent test results.

## Run the threshold tests

```sh
python -m unittest -v test_evaluation_metrics
```

These tests use synthetic scores and require NumPy and scikit-learn, but
do not require TensorFlow, the dataset, saved models, or the `outputs/` folder.

## Authors

- Mackinley Cho
- Claire Lewis
- Reese Haag
- Neha Keshan — Faculty advisor

## Acknowledgments

Thank you to our teaching assistants, Anirban Acharya and Jade Franklin,
and to our CSCI 4150 classmates for their feedback during project activities.