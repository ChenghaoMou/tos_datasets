from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import datasets
from git import Repo

from tos_datasets.proto import (
    Classification,
    Document,
    DocumentClassification,
)


@contextmanager
def download(
    repo: str = "https://github.com/SmartDataAnalytics/Polisis_Benchmark",
    keep_cache: bool = True,
    cache_dir: Path = Path.home() / ".cache" / "Polisis",
):
    repo_path = cache_dir
    if not repo_path.exists():
        Repo.clone_from(repo, repo_path)
    yield repo_path / "datasets"
    if not keep_cache:
        repo_path.unlink()


def load_data(dir: Path) -> dict[str, list[str]]:
    folders = ["Majority", "Union"]
    results: dict[str, list[str]] = defaultdict(list)

    for folder in folders:
        for file in ["train_dataset.csv", "validation_dataset.csv", "test_dataset.csv"]:
            split = file.split("_")[0]
            sentence2labels = defaultdict(set)
            df = pd.read_csv(dir / folder / file, names=["sentence", "label"])
            for _, row in df.iterrows():
                sentence2labels[row["sentence"]].add(row["label"])

            for sentence, labels in sentence2labels.items():
                results[split].append(
                    DocumentClassification(
                        document=Document(title="na", text=sentence),
                        classifications=[
                            Classification(
                                level=f"document-{folder}", labels=list(labels)
                            ),
                        ],
                    ).model_dump_json()
                )

    return results


if __name__ == "__main__":
    import pandas as pd
    import typer
    from rich import print

    def main(
        cache_dir: Path = Path.home() / ".cache" / "Polisis",
        push_to_hub: bool = False,
        keep_cache: bool = True,
    ):
        with download(keep_cache=keep_cache, cache_dir=cache_dir) as repo_path:
            data = load_data(repo_path)

        dataset = datasets.DatasetDict(
            {
                split: datasets.Dataset.from_pandas(
                    pd.DataFrame(data[split], columns=["document"])
                )
                for split in data
            }
        )
        print(
            DocumentClassification.model_validate_json(dataset["test"]["document"][0])
        )

        if push_to_hub:
            dataset.push_to_hub("chenghao/tos_pp_dataset", "privacy_glue/polisis")

    typer.run(main)
