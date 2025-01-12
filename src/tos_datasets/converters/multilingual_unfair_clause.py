# https://github.com/nlp-unibo/Multilingual-Unfair-Clause-Detection

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import datasets
from git import Repo

from tos_datasets.proto import Classification, Document, DocumentClassification


@contextmanager
def download(
    repo: str = "https://github.com/nlp-unibo/Multilingual-Unfair-Clause-Detection",
    keep_cache: bool = True,
    cache_dir: Path = Path.home() / ".cache" / "multilingual_unfair_clause",
):
    repo_path = cache_dir
    if not repo_path.exists():
        Repo.clone_from(repo, repo_path)
    yield repo_path
    if not keep_cache:
        repo_path.unlink()


def load_tags(repo_path: Path):
    tags = repo_path / "corpus" / "list_tags.txt"
    with open(tags, "r") as f:
        tags = f.readlines()
        tags = [tag.strip() for tag in tags if tag.strip()]
    return tags

def tag_definition(tag: str):

    #     The existing annotations identify nine different
    # categories for clause unfairness establishing: (1) jurisdiction for disputes in a coun-
    # try different than consumer’s residence (<j>); (2) choice of a foreign law governing
    # the contract (<law>); (3) limitation of liability (<ltd>); (4) the provider’s right to
    # unilaterally terminate the contract/access to the service (<ter>); and (5) the provid-
    # er’s right to unilaterally modify the contract/the service (<ch>); (6) requiring a con-
    # sumer to undertake arbitration before the court proceedings can commence (<a>);
    # (7) the provider retaining the right to unilaterally remove consumer content from
    # the service, including in-app purchases (<cr>); (8) having a consumer accept the
    # agreement simply by using the service, not only without reading it, but even without
    # having to click on “I agree/I accept” (<use>); (9) the scope of consent granted to the
    # ToS also takes in the privacy policy, which forms part of the “General Agreement”
    # (<pinc>). In the annotations, to indicate the degree of unfairness, a numeric value
    # was appended to each XML tag, with a value 1 meaning clearly fair, 2 potentially
    # unfair, and 3 clearly unfair.
    if not tag[-1].isdigit():
        return None

    name, score = tag[:-1], int(tag[-1])
    return ({
        "j": "jurisdiction for disputes in a country different than consumer’s residence",
        "law": "choice of a foreign law governing the contract",
        "ltd": "limitation of liability",
        "ter": "the provider’s right to unilaterally terminate the contract/access to the service",
        "ch": "the provider’s right to unilaterally modify the contract/the service",
        "a": "requiring a consumer to undertake arbitration before the court proceedings can commence",
        "cr": "the provider retaining the right to unilaterally remove consumer content from the service, including in-app purchases",
        "use": "having a consumer accept the agreement simply by using the service, not only without reading it, but even without having to click on “I agree/I accept”",
        "pinc": "the scope of consent granted to the ToS also takes in the privacy policy, which forms part of the “General Agreement”",
    }[name], {
        1: "clearly fair",
        2: "potentially unfair",
        3: "clearly unfair",
    }[score])


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
                curr_tags = [
                    t
                    for t in
                    annotation.strip().split(" ") if t.strip()
                    if tag_definition(t)
                ]
                clauses.append(Classification(
                    level="sentence",
                    labels=curr_tags,
                    label_definitions=[tag_definition(t) for t in curr_tags],
                ))
            yield DocumentClassification(
                document=doc, classifications=clauses
            ).model_dump_json()


if __name__ == "__main__":
    import pandas as pd
    import typer
    from rich import print

    from tos_datasets.proto import DocumentClassification

    def main(
        cache_dir: Path = Path.home() / ".cache" / "multilingual_unfair_clause",
        push_to_hub: bool = False,
        keep_cache: bool = True,
    ):
        with download(keep_cache=keep_cache, cache_dir=cache_dir) as repo_path:
            records = load_clauses(repo_path)

        df = pd.DataFrame(records)
        dataset = datasets.Dataset.from_pandas(df)
        dataset = dataset.rename_column("0", "document")

        print(DocumentClassification.model_validate_json(dataset["document"][0]))

        if push_to_hub:
            dataset.push_to_hub("chenghao/tos_pp_dataset", "multilingual_unfair_clause")

    typer.run(main)
