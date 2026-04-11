import argparse
import csv
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


VALID_SENTIMENTS = {"positive", "negative", "neutral"}
DATASET_URLS = {
    "laptop": "https://huggingface.co/datasets/alexcadillon/SemEval2014Task4/raw/main/SemEval%2714-ABSA-TrainData_v2%20%26%20AnnotationGuidelines/Laptop_Train_v2.xml",
    "restaurant": "https://huggingface.co/datasets/alexcadillon/SemEval2014Task4/raw/main/SemEval%2714-ABSA-TrainData_v2%20%26%20AnnotationGuidelines/Restaurants_Train_v2.xml",
}


def normalize_aspect(text: str) -> str:
    return text.strip().lower().replace("/", "_").replace("#", "_").replace(" ", "_")


def download_xml(dataset: str, output_path: Path) -> Path:
    url = DATASET_URLS[dataset]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        output_path.write_bytes(response.read())
    return output_path


def parse_aspect_terms(xml_path: Path) -> list[dict[str, str]]:
    root = ET.parse(xml_path).getroot()
    rows = []
    for sentence in root.iter("sentence"):
        sid = sentence.attrib.get("id", "")
        text = sentence.findtext("text", default="").strip()
        terms_node = sentence.find("aspectTerms")
        if not sid or not text or terms_node is None:
            continue
        for term in terms_node.findall("aspectTerm"):
            polarity = term.attrib.get("polarity", "").strip().lower()
            aspect = normalize_aspect(term.attrib.get("term", ""))
            if polarity not in VALID_SENTIMENTS or not aspect:
                continue
            rows.append({
                "id": sid,
                "review": text,
                "aspect": aspect,
                "sentiment": polarity,
            })
    return rows


def save_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "review", "aspect", "sentiment"])
        writer.writeheader()
        writer.writerows(rows)


def print_stats(rows: list[dict[str, str]], dataset: str) -> None:
    print(f"Dataset: {dataset}")
    print(f"Rows: {len(rows)}")
    print(f"Unique review ids: {len({row['id'] for row in rows})}")
    print(f"Sentiment distribution: {dict(Counter(row['sentiment'] for row in rows))}")
    print(f"Top aspects: {dict(Counter(row['aspect'] for row in rows).most_common(15))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert SemEval 2014 ABSA data.")
    parser.add_argument("--dataset", choices=sorted(DATASET_URLS), required=True)
    parser.add_argument("--xml", type=Path, help="Existing XML file path. If omitted, the file is downloaded.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data") / "raw_semeval",
        help="Directory used when downloading XML files.",
    )
    args = parser.parse_args()

    xml_path = args.xml
    if xml_path is None:
        xml_path = args.raw_dir / f"{args.dataset}.xml"
        download_xml(args.dataset, xml_path)

    rows = parse_aspect_terms(xml_path)
    save_csv(rows, args.output)
    print_stats(rows, args.dataset)


if __name__ == "__main__":
    main()
