# https://github.com/nlp-unibo/Multilingual-Unfair-Clause-Detection

from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import datasets
from git import Repo

from tos_datasets.proto import Classification, Document, DocumentClassification


@contextmanager
def download(
    repo: str = "https://github.com/federicoruggeri/Memnet_ToS",
    keep_cache: bool = True,
    cache_dir: Path = Path.home() / ".cache" / "memnet_tos",
):
    repo_path = cache_dir
    if not repo_path.exists():
        Repo.clone_from(repo, repo_path)
    yield repo_path
    if not keep_cache:
        repo_path.unlink()


def load_tags(repo_path: Path):
    tags = (repo_path / "local_database" / "KB").glob("*KB.txt")
    results = defaultdict(dict)
    for tag_file in tags:
        category = tag_file.name.split("_")[0]
        results[category] = {
            i: line for i, line in enumerate(Path(tag_file).read_text().splitlines())
        }
    return results


def load_clauses(repo_path: Path, tags: dict) -> Generator[str, None, None]:
    corpus_path = repo_path / "local_database" / "ToS_100" / "dataset.csv"
    df = pd.read_csv(corpus_path, index_col=0)
    for doc_id, group in df.groupby("document_ID"):
        sentences = group["text"].tolist()

        doc = Document(
            title=group["document"].iloc[0],
            language="en",
            sentences=sentences,
        )
        clauses = []
        for sentence, (_, annotation) in zip(sentences, group.iterrows()):
            # A	CH	CR	J	LAW	LTD	PINC	TER	USE
            # TER_targets	LTD_targets	A_targets	CH_targets	CR_targets
            curr_tags = []
            curr_tag_definitions = []
            for tag in ["A", "CH", "CR", "J", "LAW", "LTD", "PINC", "TER", "USE"]:
                if annotation[tag] == 1:
                    curr_tags.append(tag)
                if f"{tag}_targets" in annotation:
                    ids = annotation[f"{tag}_targets"]
                    if pd.isna(ids):
                        continue
                    curr_tag_definitions.append(
                        [tags[tag][int(id)] for id in ids.strip("[]").split(",")]
                    )

            clauses.append(
                Classification(
                    level="sentence",
                    labels=curr_tags,
                    label_definitions=curr_tag_definitions,
                )
            )
        yield DocumentClassification(
            document=doc, classifications=clauses
        ).model_dump_json()


if __name__ == "__main__":
    import pandas as pd
    import typer
    from rich import print

    from tos_datasets.proto import DocumentClassification

    def main(
        cache_dir: Path = Path.home() / ".cache" / "memnet_tos",
        push_to_hub: bool = False,
        keep_cache: bool = True,
    ):
        with download(keep_cache=keep_cache, cache_dir=cache_dir) as repo_path:
            tags = load_tags(repo_path)
            records = load_clauses(repo_path, tags)

        df = pd.DataFrame(records)
        dataset = datasets.Dataset.from_pandas(df)
        dataset = dataset.rename_column("0", "document")

        print(DocumentClassification.model_validate_json(dataset["document"][0]))

        if push_to_hub:
            dataset.push_to_hub("chenghao/tos_pp_dataset", "memnet_tos")

    typer.run(main)
