# https://github.com/nlp-unibo/Multilingual-Unfair-Clause-Detection

from pathlib import Path
from typing import Generator

import datasets
from git import Repo

from tos_datasets.proto import Classification, Document, DocumentClassification


def download(
    repo: str = "https://github.com/nlp-unibo/Multilingual-Unfair-Clause-Detection",
):
    repo_path = Path.home() / ".cache" / repo.split("/")[-1]
    if not repo_path.exists():
        Repo.clone_from(repo, repo_path)
    return repo_path


def load_tags(repo_path: Path):
    tags = repo_path / "corpus" / "list_tags.txt"
    with open(tags, "r") as f:
        tags = f.readlines()
        tags = [tag.strip() for tag in tags if tag.strip()]
    return tags


def load_clauses(repo_path: Path) -> Generator[str, None, None]:
    corpus_path = repo_path / "corpus"

    for lan in (corpus_path / "sentences").iterdir():
        if not lan.is_dir():
            continue
        for file in (lan / "original").iterdir():
            if not file.is_file():
                continue
            full_text = file.read_text()
            sentences = full_text.splitlines()
            annotations = corpus_path / "tags" / lan.name / "original" / file.name
            if not annotations.is_file():
                continue
            annotations = annotations.read_text().splitlines()
            assert len(sentences) == len(annotations)
            doc = Document(
                title=file.name,
                language=lan.name,
                sentences=sentences,
                text=full_text,
            )
            clauses = []
            for sentence, annotation in zip(sentences, annotations):
                curr_tags = [t for t in annotation.strip().split(" ") if t.strip()]
                clauses.append(Classification(level="sentence", labels=curr_tags))
            yield DocumentClassification(
                document=doc, classifications=clauses
            ).model_dump_json()


if __name__ == "__main__":
    import pandas as pd

    repo_path = download()
    records = load_clauses(repo_path)
    df = pd.DataFrame(records)
    dataset = datasets.Dataset.from_pandas(df)
    dataset = dataset.rename_column("0", "document")
    dataset.push_to_hub("chenghao/tos_pp_dataset", "multilingual_unfair_clause")
