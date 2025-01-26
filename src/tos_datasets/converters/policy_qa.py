import json
from contextlib import contextmanager
from pathlib import Path

import datasets
from git import Repo

from tos_datasets.proto import (
    QA,
    Document,
    DocumentQA,
)


@contextmanager
def download(
    repo: str = "https://github.com/wasiahmad/PolicyQA",
    keep_cache: bool = True,
    cache_dir: Path = Path.home() / ".cache" / "PolicyQA",
):
    repo_path = cache_dir
    if not repo_path.exists():
        Repo.clone_from(repo, repo_path)
    yield repo_path
    if not keep_cache:
        repo_path.unlink()


def load_data(repo_path: Path) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    for file in repo_path.glob("data/*.json"):
        split = file.stem
        results[split] = []
        with open(file, "r") as f:
            data = json.load(f)

            for record in data["data"]:
                title = record["title"]
                paragraph_text = []
                doc_text = ""
                annotations = []
                paragraphs = record["paragraphs"]
                for paragraph in paragraphs:
                    context = paragraph["context"]
                    context += "\n"
                    qas = paragraph["qas"]
                    for qa in qas:
                        question = qa["question"]
                        answer = qa["answers"][0]["text"]
                        answer_start = qa["answers"][0]["answer_start"]
                        global_start = len(doc_text) + answer_start
                        global_end = global_start + len(answer)

                        annotations.append(
                            QA(
                                question=question,
                                answer=answer,
                                start=global_start,
                                end=global_end,
                                is_impossible=False,
                            )
                        )

                    doc_text += context
                    paragraph_text.append(context)

                doc = Document(title=title, text=doc_text, paragraphs=paragraph_text)
                doc_qa = DocumentQA(document=doc, qas=annotations)
                results[split].append(doc_qa.model_dump_json())

    return results


if __name__ == "__main__":
    import pandas as pd
    import typer
    from rich import print

    def main(
        cache_dir: Path = Path.home() / ".cache" / "PolicyQA",
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

        print(DocumentQA.model_validate_json(dataset["train"]["document"][0]))

        if push_to_hub:
            dataset.push_to_hub("chenghao/tos_pp_dataset", "privacy_glue/policy_qa")

    typer.run(main)
