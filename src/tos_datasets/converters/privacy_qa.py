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
    repo: str = "https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP",
    keep_cache: bool = True,
    cache_dir: Path = Path.home() / ".cache" / "PrivacyQA",
):
    repo_path = cache_dir
    if not repo_path.exists():
        Repo.clone_from(repo, repo_path)
    yield repo_path / "data"
    if not keep_cache:
        repo_path.unlink()


def load_data(dir: Path) -> dict[str, list[str]]:
    results: dict[str, list[str]] = defaultdict(list)

    for file in ["policy_train_data.csv", "policy_test_data.csv"]:
        split = file.split("_")[1]
        df = pd.read_csv(dir / file, sep="\t")
        label_col = "Any_Relevant" if split == "test" else "Label"
        for doc_id, group in df.groupby("DocID"):
            classifications = []
            segments = []
            for query, rows in group.groupby("QueryID"):
                classifications.append(
                    Classification(
                        level="sentence",
                        labels=list(rows[label_col].tolist()),
                        label_definitions=[[q] for q in rows["Query"]],
                    )
                )
                if not segments:
                    segments = rows["Segment"].tolist()

            results[split].append(
                DocumentClassification(
                    document=Document(title=doc_id, sentences=segments),
                    classifications=classifications,
                ).model_dump_json()
            )

    return results


if __name__ == "__main__":
    import pandas as pd
    import typer
    from rich import print

    def main(
        cache_dir: Path = Path.home() / ".cache" / "PrivacyQA",
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
            dataset.push_to_hub("chenghao/tos_pp_dataset", "privacy_glue/privacy_qa")

    typer.run(main)
