import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def read_predictions(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"fold", "y", "p_malignant"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Predictions CSV must contain fold, y, and p_malignant.")

        rows = []
        for line, row in enumerate(reader, start=2):
            label = float(row["y"])
            score = float(row["p_malignant"])
            if label not in (0, 1) or not 0 <= score <= 1:
                raise ValueError(f"Invalid label or score on CSV line {line}.")
            rows.append((int(row["fold"]), int(label), score))

    if {label for _, label, _ in rows} != {0, 1}:
        raise ValueError("Predictions must include both benign and malignant records.")
    return rows


def metrics(rows, cutoff):
    tp = fp = tn = fn = 0
    for _, label, score in rows:
        flagged = score >= cutoff
        if label == 1:
            tp += int(flagged)
            fn += int(not flagged)
        else:
            fp += int(flagged)
            tn += int(not flagged)

    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return recall, specificity, fp, fn


def percent(value):
    return "N/A" if value is None else f"{value:.1%}"


def main():
    parser = argparse.ArgumentParser(description="Compare saved and corrected cutoffs.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("threshold_json", type=Path)
    parser.add_argument("--output", type=Path, default=Path("fold_metrics_corrected.csv"))
    args = parser.parse_args()

    try:
        rows = read_predictions(args.predictions)
        config = json.loads(args.threshold_json.read_text(encoding="utf-8-sig"))
        saved = float(config["threshold"])
        target = float(config["target_sensitivity"])
        if not 0 <= saved <= 1 or not 0 < target <= 1:
            raise ValueError("Expected threshold in [0, 1] and target_sensitivity in (0, 1].")

        malignant_scores = sorted(
            (score for _, label, score in rows if label == 1), reverse=True
        )
        # The lowest score among the required true positives is the
        # highest cutoff that still catches at least that many.
        required_hits = math.ceil(target * len(malignant_scores))
        corrected = malignant_scores[required_hits - 1]

        folds = defaultdict(list)
        for row in rows:
            folds[row[0]].append(row)

        # Exclusive creation keeps an existing report from being overwritten.
        with args.output.open("x", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["fold", "pat_sens@thr", "pat_spec@thr"])
            for fold, fold_rows in sorted(folds.items()):
                recall, specificity, _, _ = metrics(fold_rows, corrected)
                writer.writerow([fold, recall, specificity])

    except FileExistsError:
        parser.exit(1, f"{args.output} already exists. Use --output with a new filename.\n")
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        parser.exit(1, f"Could not compare thresholds: {error}\n")

    print(f"Malignant recall target: {target:.1%}")
    for name, cutoff in [("Saved", saved), ("Corrected", corrected)]:
        recall, specificity, fp, fn = metrics(rows, cutoff)
        print(f"{name:9} cutoff={cutoff:.5f} | recall={percent(recall)} | "
              f"specificity={percent(specificity)} | false alarms={fp} | misses={fn}")

    print(f"\nSaved {args.output}")
    print("Exploratory comparison: the same predictions were used to choose "
          "and score the corrected cutoff. This is not an independent test.")


if __name__ == "__main__":
    main()