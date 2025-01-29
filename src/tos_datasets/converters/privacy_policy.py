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
    repo: str = "https://github.com/infsys-lab/policy-detection-data",
    keep_cache: bool = True,
    cache_dir: Path = Path.home() / ".cache" / "PrivacyPolicy",
):
    repo_path = cache_dir
    if not repo_path.exists():
        Repo.clone_from(repo, repo_path)
    yield repo_path / "data" / "1301_dataset.csv.xz"
    if not keep_cache:
        repo_path.unlink()


def load_data(file_path: Path) -> Generator[str, None, None]:
    df = pd.read_csv(file_path, index_col=0)
    for _, row in df.iterrows():
        yield DocumentClassification(
            document=Document(title=row["link_text"], text=row["policy_text"]),
            classifications=[
                Classification(
                    level="document",
                    labels=["is_policy" if row["is_policy"] else "not_policy"],
                )
            ],
        ).model_dump_json()


if __name__ == "__main__":
    import pandas as pd
    import typer
    from rich import print

    def main(
        cache_dir: Path = Path.home() / ".cache" / "PrivacyPolicy",
        push_to_hub: bool = False,
        keep_cache: bool = True,
    ):
        with download(keep_cache=keep_cache, cache_dir=cache_dir) as repo_path:
            data = load_data(repo_path)

        ds = datasets.Dataset.from_pandas(pd.DataFrame(data, columns=["document"]))

        print(DocumentClassification.model_validate_json(ds["document"][0]))

        if push_to_hub:
            ds.push_to_hub("chenghao/tos_pp_dataset", "privacy_glue/policy_detection")

    typer.run(main)
