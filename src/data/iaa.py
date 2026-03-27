# src/data/iaa.py
import csv
from sklearn.metrics import cohen_kappa_score


def load_annotations(path_a: str, path_b: str) -> tuple[list[str], list[str]]:
    """Load two annotation CSV files and return aligned sentiment lists."""
    def read_csv(path):
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return {row["id"]: row["sentiment"] for row in reader}

    ann_a = read_csv(path_a)
    ann_b = read_csv(path_b)
    common_ids = sorted(set(ann_a) & set(ann_b))
    labels_a = [ann_a[i] for i in common_ids]
    labels_b = [ann_b[i] for i in common_ids]
    return labels_a, labels_b


def compute_cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute Cohen's kappa between two annotators."""
    return cohen_kappa_score(labels_a, labels_b)


def iaa_report(path_a: str, path_b: str) -> dict:
    """Return a full IAA report dict."""
    labels_a, labels_b = load_annotations(path_a, path_b)
    kappa = compute_cohens_kappa(labels_a, labels_b)
    agreed = sum(a == b for a, b in zip(labels_a, labels_b))
    if len(labels_a) == 0:
        raise ValueError("No common IDs found between the two annotation files.")
    return {
        "n_samples": len(labels_a),
        "agreement_rate": agreed / len(labels_a),
        "cohens_kappa": round(kappa, 4),
        "interpretation": (
            "poor" if kappa < 0.2 else
            "fair" if kappa < 0.4 else
            "moderate" if kappa < 0.6 else
            "substantial" if kappa < 0.8 else
            "almost perfect"
        ),
    }


if __name__ == "__main__":
    import sys
    report = iaa_report(sys.argv[1], sys.argv[2])
    for k, v in report.items():
        print(f"{k}: {v}")
