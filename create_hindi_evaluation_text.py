"""Create a Hindi evaluation text file from the FLORES+ devtest split."""

import argparse
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evaluation_text.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(
        "openlanguagedata/flores_plus",
        "hin_Deva",
        split="devtest",
    )

    articles: dict[str, list[str]] = defaultdict(list)
    for index, row in enumerate(dataset):
        text = " ".join(row["text"].split())
        source = row["url"] or f"sentence-{index}"
        if text:
            articles[source].append(text)

    documents = [" ".join(sentences) for sentences in articles.values()]
    output_path = Path(args.output)
    output_path.write_text("\n\n".join(documents) + "\n", encoding="utf-8")

    print(f"Saved {len(documents)} documents to {output_path}")


if __name__ == "__main__":
    main()
