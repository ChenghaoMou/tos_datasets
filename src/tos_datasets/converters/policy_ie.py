import json
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from itertools import groupby
from pathlib import Path

import datasets
from git import Repo

from tos_datasets.proto import (
    Document,
    DocumentEvent,
    DocumentSequenceClassification,
    Event,
    Tag,
)


@contextmanager
def download(
    repo: str = "https://github.com/infsys-lab/policy-ie",
    keep_cache: bool = True,
    cache_dir: Path = Path.home() / ".cache" / "PolicyIE",
):
    repo_path = cache_dir
    if not repo_path.exists():
        Repo.clone_from(repo, repo_path)
    # unzip data/sanitized_split.zip
    with zipfile.ZipFile(repo_path / "data/sanitized_split.zip", "r") as zip_ref:
        zip_ref.extractall(repo_path / "data")
    yield repo_path / "data" / "sanitized_split"
    if not keep_cache:
        repo_path.unlink()


def load_data(repo_path: Path) -> dict[str, list[str]]:
    results: dict[str, list[tuple[str, str]]] = defaultdict(list)
    # unzip data/sanitized_split.zip

    for p, files in groupby(repo_path.glob("**/*.json"), key=lambda x: x.parent):
        split = "train" if "train" in str(p) else "test"
        paragraphs = []
        spans = []
        events = []
        global_start = 0
        for f in sorted(files, key=lambda x: int(x.stem)):
            with open(f) as inp:
                data = json.load(inp)
                text: str = data["text"]
                paragraphs.append(text)
                for entity in data["entity/argument_mentions"]:
                    label: str = entity["entity/argument_type"]
                    start: int = entity["start_idx"]
                    end: int = entity["end_idx"]
                    assert text[start:end] == entity["entity/argument_text"]
                    spans.append(
                        Tag(
                            tag=label,
                            start=global_start + start,
                            end=global_start + end,
                            comment="entity/argument_mentions",
                        )
                    )

                for event in data["event_mentions"]:
                    event_type = event["event_type"]
                    trigger = Tag(
                        tag=f"""{event_type} - trigger""",
                        start=global_start + event["trigger"]["start_idx"],
                        end=global_start + event["trigger"]["end_idx"],
                    )
                    arguments = []
                    for arg in event["arguments"]:
                        arguments.append(
                            Tag(
                                tag=arg["type"],
                                start=global_start + arg["start_idx"],
                                end=global_start + arg["end_idx"],
                                comment=f"""role: {arg["role"]}""",
                            )
                        )
                    events.append(
                        Event(
                            event_type=event_type, trigger=trigger, arguments=arguments
                        )
                    )

                global_start += len(text)

        results[split].append(
            (
                DocumentSequenceClassification(
                    document=Document(
                        title=p.name, paragraphs=paragraphs, language="en"
                    ),
                    tags=spans,
                ).model_dump_json(),
                DocumentEvent(
                    document=Document(
                        title=p.name, paragraphs=paragraphs, language="en"
                    ),
                    events=events,
                ).model_dump_json(),
            )
        )

    return results


if __name__ == "__main__":
    import pandas as pd
    import typer
    from rich import print

    def main(
        cache_dir: Path = Path.home() / ".cache" / "PolicyIE",
        push_to_hub: bool = False,
        keep_cache: bool = True,
    ):
        with download(keep_cache=keep_cache, cache_dir=cache_dir) as repo_path:
            # print(repo_path)
            data = load_data(repo_path)

        dataset = datasets.DatasetDict(
            {
                split: datasets.Dataset.from_pandas(
                    pd.DataFrame(data[split], columns=["type_i", "type_ii"])
                )
                for split in data
            }
        )

        print(
            DocumentSequenceClassification.model_validate_json(
                dataset["train"]["type_i"][0]
            )
        )
        print(DocumentEvent.model_validate_json(dataset["train"]["type_ii"][0]))

        if push_to_hub:
            dataset.push_to_hub("chenghao/tos_pp_dataset", "privacy_glue/policy_ie")

    typer.run(main)
