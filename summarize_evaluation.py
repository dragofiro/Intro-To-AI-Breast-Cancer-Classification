import argparse
import csv
import math
from pathlib import Path

METRICS = ("pat_sens@thr", "pat_spec@thr")


def read_rate(row, column):
    value = (row.get(column) or "").strip()
    if not value:
        return None
    rate = float(value)
    if math.isnan(rate):
        return None
    if not 0 <= rate <= 1:
        raise ValueError(f"Fold {row['fold']}: {column} must be between 0 and 1.")
    return rate


def percent(rate):
    return "N/A" if rate is None else f"{rate:.1%}"


def main():
    parser = argparse.ArgumentParser(description="Summarize per-fold evaluation results.")
    parser.add_argument("csv_path", nargs="?", type=Path,
                        default=Path("outputs/fold_metrics.csv"))
    args = parser.parse_args()

    try:
        with args.csv_path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            missing = {"fold", *METRICS} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
            rows = [
                (row["fold"], *(read_rate(row, column) for column in METRICS))
                for row in reader
            ]
        if not rows:
            raise ValueError("The CSV contains no fold results.")
    except FileNotFoundError:
        parser.exit(1, f"{args.csv_path} not found. Run the evaluation or pass a CSV path.\n")
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Could not read results: {error}\n")

    print(f"{'Fold':<8} {'Malignant recall':>18} {'Specificity':>14}")
    for fold, recall, specificity in rows:
        print(f"{fold:<8} {percent(recall):>18} {percent(specificity):>14}")

    recalls = [(fold, recall) for fold, recall, _ in rows if recall is not None]
    if recalls:
        fold, recall = min(recalls, key=lambda pair: pair[1])
        print(f"\nLowest malignant recall: {percent(recall)} (fold {fold}).")

    print("Malignant recall: share of malignant patient records correctly flagged.")
    print("Specificity: share of benign patient records correctly left unflagged.")
    print("These folds use a cutoff chosen from the same pooled predictions; "
          "these are not independent test results.")


if __name__ == "__main__":
    main()