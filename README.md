# Intro-To-AI-Breast-Cancer-Classification
This is the repository for the RPI course CS 4150, Intro To AI, using the BreakHis dataset from https://www.kaggle.com/datasets/ambarish/breakhis/data

## Run the EfficientNetB0 evaluation

From the repository root, check that the PNG files are under
`BreaKHis_v1/histology_slides/breast`, or update `DATA_DIR` in
`efficient_b_0.py`. With TensorFlow, NumPy, pandas, and scikit-learn
installed in your project environment, run:

```sh
python efficient_b_0.py
```

This command **trains the model across five grouped folds** and evaluates the
predictions. It creates `outputs/` locally; you do not need an existing copy.
The image cache and model checkpoints can take substantial disk space.

| Output | Contents |
| --- | --- |
| `outputs/oof_predictions.csv` | Image-level labels, fold IDs, and predicted malignancy scores. |
| `outputs/patient_predictions.csv` | Scores aggregated across images for each patient record in a fold. |
| `outputs/threshold.json` | Selected cutoff, target sensitivity, and aggregation setting. |
| `outputs/fold*_best.keras` | Saved model for each fold. |

The default target is 95% patient-level sensitivity. The script chooses the
highest cutoff that meets the target on the pooled out-of-fold predictions,
then reports sensitivity, specificity, and confusion-matrix counts. The same
predictions are used to select and score the cutoff, so these are exploratory
results, not independent test results or evidence of clinical readiness. For an
independent estimate, fix the cutoff before evaluating on a separate patient group
that was not used for training, model selection, or threshold selection.

## Verify threshold selection without training

From the repository root, with NumPy and scikit-learn installed, run:

```sh
python -m unittest -v test_evaluation_metrics
```

The tests use synthetic scores and require no dataset, saved models, or
`outputs/` folder.

## Authors

- Mackinley Cho
- Claire Lewis
- Reese Haag
- Neha Keshan — Faculty advisor

## Acknowledgments

Thank you to our teaching assistants, Anirban Acharya and Jade Franklin,
and to our CSCI 4150 classmates for their feedback during project activities.