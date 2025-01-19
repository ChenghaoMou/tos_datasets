import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pandas as pd
import requests
from loguru import logger

from tos_datasets.proto import (
    Classification,
    Document,
    DocumentClassification,
)


@contextmanager
def download_and_unzip(
    url: str = "http://claudette.eui.eu/corpus_142_ToS.zip",
    cache_dir: Path = Path.home() / ".cache" / "142_tos",
    keep_cache: bool = True,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "142_tos.zip"

    if not zip_path.exists():
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    extract_dir = cache_dir / "corpus"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(cache_dir)

    yield extract_dir

    if not keep_cache:
        cache_dir.unlink()


def load_annotations(
    local_dir: Path,
) -> Generator[tuple[Document, list[str]], None, None]:
    for file in (local_dir / "sentences").glob("*.txt"):
        company = file.name.replace(".txt", "")
        doc = Path(file).read_text()
        lines = doc.splitlines()
        anno = Path(str(file).replace("/sentences/", "/tags_unfair/")).read_text()
        anno = anno.splitlines()
        if len(lines) != len(anno):
            logger.warning(f"{len(lines)} != {len(anno)}")
            continue
        yield (
            Document(
                title=company,
                language="en",
                sentences=lines,
                text=doc,
            ),
            anno,
        )


def load_definitions(local_dir: Path) -> dict[str, tuple[str, str]]:
    tags = Path(local_dir / "lists" / "list_tags.txt").read_text().splitlines()
    results = {}
    for tag in tags:
        name, score = tag[:-1], int(tag[-1])
        results[tag] = (
            {
                "j": "jurisdiction for disputes in a country different than consumer’s residence",
                "law": "choice of a foreign law governing the contract",
                "ltd": "limitation of liability",
                "ter": "the provider’s right to unilaterally terminate the contract/access to the service",
                "ch": "the provider’s right to unilaterally modify the contract/the service",
                "a": "requiring a consumer to undertake arbitration before the court proceedings can commence",
                "cr": "the provider retaining the right to unilaterally remove consumer content from the service, including in-app purchases",
                "use": "having a consumer accept the agreement simply by using the service, not only without reading it, but even without having to click on “I agree/I accept”",
                "pinc": "the scope of consent granted to the ToS also takes in the privacy policy, which forms part of the “General Agreement”",
            }[name],
            {
                1: "clearly fair",
                2: "potentially unfair",
                3: "clearly unfair",
            }[score],
        )
    return results


def convert(
    annotations: Generator[tuple[Document, list[str]], None, None],
    definitions: dict[str, tuple[str, str]],
) -> Generator[str, None, None]:
    for doc, anno in annotations:
        clauses = []
        for sentence, annotation in zip(doc.sentences, anno):
            curr_tags = [
                t
                for t in annotation.strip().split(" ")
                if t.strip()
                if t in definitions
            ]
            clauses.append(
                Classification(
                    level="sentence",
                    labels=curr_tags,
                    label_definitions=[definitions[t] for t in curr_tags],
                )
            )
        yield DocumentClassification(
            document=doc,
            classifications=clauses,
        ).model_dump_json()


if __name__ == "__main__":
    import datasets
    import typer
    from rich import print

    from tos_datasets.proto import DocumentClassification

    def main(
        push_to_hub: bool = False,
        keep_cache: bool = True,
        cache_dir: Path = Path.home() / ".cache" / "142_tos",
    ):
        with download_and_unzip(
            cache_dir=cache_dir, keep_cache=keep_cache
        ) as local_dir:
            annotations = load_annotations(local_dir)
            definitions = load_definitions(local_dir)

        df = pd.DataFrame(list(convert(annotations, definitions)))
        ds = datasets.Dataset.from_pandas(df)
        ds = ds.rename_column("0", "document")

        print(DocumentClassification.model_validate_json(ds["document"][0]))

        if push_to_hub:
            ds.push_to_hub("chenghao/tos_pp_dataset", "142_tos")

    typer.run(main)
