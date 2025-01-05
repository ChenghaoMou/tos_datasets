import zipfile
from pathlib import Path
from typing import Generator

import fitz
import pandas as pd
import requests
from loguru import logger

from tos_datasets.proto import (
    Document,
    DocumentEUConsumerLawAnnotation,
    EUConsumerLawAnnotation,
    Service,
)


def download_and_unzip(
    url: str = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/dtbj87j937-3.zip",
):
    cache_dir = Path.home() / ".cache" / "100_tos"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "100_tos.zip"

    if not zip_path.exists():
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    extract_dir = cache_dir / "Annotated Terms of Service of 100 Online Platforms"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(cache_dir)

    return extract_dir


def load_annotations(local_dir: Path) -> pd.DataFrame:
    annotations = pd.read_csv(
        local_dir / "Terms of Service Analysis and Evaluation_RESULTS.csv", sep=";"
    )
    annotations = annotations.assign(full_text=["" for _ in range(len(annotations))])
    companies = sorted(annotations.name.str.lower().unique())
    for file in (local_dir / "Clear ToS").glob("*.pdf"):
        company = file.name.replace(".pdf", "")
        if company.lower() not in companies:
            continue
        doc = fitz.open(file)
        text = "\n".join(page.get_text() for page in doc)
        annotations.loc[
            annotations.name.str.lower() == company.lower(), "full_text"
        ] = text
    return annotations


def load_definitions(local_dir: Path) -> pd.DataFrame:
    definitions = pd.read_excel(local_dir / "Variables Definitions.xlsx")
    return definitions


def convert(
    annotations: pd.DataFrame,
    definitions: pd.DataFrame,
) -> Generator[DocumentEUConsumerLawAnnotation, None, None]:
    definitions = definitions.ffill()
    labels = {}
    for start in range(0, len(definitions), 3):
        group = definitions.iloc[start : start + 3]
        if group["Score"].nunique() != 3:
            continue
        for _, row in group.iterrows():
            labels[(row["Code"], row["Score"])] = EUConsumerLawAnnotation(
                general_category=row["General category"],
                name=row["Variable name"],
                legal_ground=row["Legal ground"],
                code=row["Code"],
                score=row["Score"],
                explanation=row["Detailed description"],
            )

    for record in annotations.to_dict(orient="records"):
        if record["full_text"] == "":
            continue
        try:
            date = record["date"] if not pd.isna(record["date"]) else ""
            yield DocumentEUConsumerLawAnnotation(
                service=Service(
                    name=record["name"],
                    url=record["url"],
                    date=date,
                    lang=record["lang"],
                    sector=record["sector"],
                    hq=record["hq"],
                    hq_category=record["hq_cat"],
                    is_public=record["public"],
                    is_paid=record["paid"],
                ),
                document=Document(
                    title="",
                    text=record["full_text"],
                ),
                annotations=[
                    labels[(key, value)]
                    for key, value in record.items()
                    if (key, value) in labels
                ],
            ).model_dump_json()
        except Exception as e:
            logger.error(f"Error processing record: {e}")
            continue


if __name__ == "__main__":
    import datasets

    local_dir = download_and_unzip()
    logger.info(f"Downloaded and unzipped to {local_dir}")
    annotations = load_annotations(local_dir)
    definitions = load_definitions(local_dir)

    df = pd.DataFrame(list(convert(annotations, definitions)))
    ds = datasets.Dataset.from_pandas(df)
    ds = ds.rename_column("0", "document")
    ds.push_to_hub("chenghao/tos_pp_dataset", "100_tos")
