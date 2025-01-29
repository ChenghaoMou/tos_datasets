from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import datasets
from git import Repo
from nltk.tokenize.treebank import TreebankWordDetokenizer

from tos_datasets.proto import (
    Document,
    DocumentSequenceClassification,
    Tag,
)

detokenizer = TreebankWordDetokenizer()


@contextmanager
def download(
    repo: str = "https://github.com/um-rtcl/piextract_dataset",
    keep_cache: bool = True,
    cache_dir: Path = Path.home() / ".cache" / "Piextract",
):
    repo_path = cache_dir
    if not repo_path.exists():
        Repo.clone_from(repo, repo_path)
    yield repo_path / "dataset"
    if not keep_cache:
        repo_path.unlink()


def load_data(repo_path: Path) -> dict[str, list[str]]:
    results: dict[str, list[str]] = defaultdict(list)
    # unzip data/sanitized_split.zip

    for f in repo_path.glob("**/*.conll03"):
        split = f.stem
        with open(f, "r") as file:
            for sentence in file.read().split("\n\n"):
                tokens = []
                labels = []
                for line in sentence.split("\n"):
                    if line.count(" ") != 3:
                        continue
                    token, _, _, tag = line.split(" ")
                    if token == "-DOCSTART-":
                        continue
                    tokens.append(token)
                    labels.append(tag)

                if not tokens:
                    continue
                text = detokenizer.detokenize(tokens)
                start = 0
                tags = []
                for token, label in zip(tokens, labels):
                    while text[start : start + len(token)] != token:
                        start += 1
                    tags.append(
                        Tag(
                            tag=label,
                            start=start,
                            end=start + len(token),
                        )
                    )
                results[split].append(
                    DocumentSequenceClassification(
                        document=Document(title="na", text=text, tokens=tokens),
                        tags=tags,
                    ).model_dump_json()
                )
                # results[split].append(sentence.text)
    return results


if __name__ == "__main__":
    import pandas as pd
    import typer
    from rich import print

    def main(
        cache_dir: Path = Path.home() / ".cache" / "Piextract",
        push_to_hub: bool = False,
        keep_cache: bool = True,
    ):
        with download(keep_cache=keep_cache, cache_dir=cache_dir) as repo_path:
            # print(repo_path)
            data = load_data(repo_path)
            print(data["validation"][0])

        dataset = datasets.DatasetDict(
            {
                split: datasets.Dataset.from_pandas(
                    pd.DataFrame(data[split], columns=["document"])
                )
                for split in data
            }
        )

        print(
            DocumentSequenceClassification.model_validate_json(
                dataset["train"]["document"][0]
            )
        )
        # print(DocumentEvent.model_validate_json(dataset["train"]["type_ii"][0]))

        if push_to_hub:
            dataset.push_to_hub("chenghao/tos_pp_dataset", "privacy_glue/piextract")

    typer.run(main)
