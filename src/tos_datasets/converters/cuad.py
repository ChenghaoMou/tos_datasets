import json
import zipfile
from pathlib import Path

import datasets
import pandas as pd
import requests

from tos_datasets.proto import QA, Document, DocumentQA


def download_and_unzip(
    url: str = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1",
):
    cache_dir = Path.home() / ".cache" / "cuad"
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = cache_dir / "CUAD_v1.zip"

    if not zip_path.exists():
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    extract_dir = cache_dir / "CUAD_v1"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(cache_dir)

    return extract_dir


def load_annotations(local_dir: Path):
    with open(local_dir / "CUAD_v1.json") as f:
        data = json.load(f)
    annotations = data["data"]
    return annotations


def clean_cache():
    (Path.home() / ".cache" / "cuad").unlink()


def collect_target_files(target: str = "Service"):
    for part in ["I", "II", "III"]:
        for file in (local_dir / "full_contract_pdf" / f"Part_{part}" / target).glob(
            "*.*"
        ):
            basename = file.name
            txt_file = list((local_dir / "full_contract_txt").glob(f"{basename[:-3]}*"))
            if not txt_file:
                continue
            yield {"pdf_path": file, "text": Path(txt_file[0]).read_text()}


def annotate(target_files, annotations):
    for file in target_files:
        name = file["pdf_path"].stem
        full_text = file["text"]

        doc = DocumentQA(document=Document(title=name, text=full_text), qas=[])

        for anno in annotations:
            if anno["title"].lower() != name.lower():
                continue
            for paragraph in anno["paragraphs"]:
                context = paragraph["context"]
                # assert context == full_text
                for qa in paragraph["qas"]:
                    category = qa["id"].rsplit("__", 1)[-1]
                    is_impossible = qa["is_impossible"]
                    for answer in qa["answers"]:
                        text = answer["text"]
                        start = answer["answer_start"]
                        end = start + len(text)
                        assert (
                            context[start:end] == text
                        ), f"{context[start:end]} != {text}"
                        # labels.append(
                        #     {
                        #         "start": start,
                        #         "end": end,
                        #         "text": text,
                        #         "label": category,
                        #         "impossible": is_impossible,
                        #     }
                        # )
                        doc.qas.append(
                            QA(
                                question=category,
                                answer=text,
                                start=start,
                                end=end,
                                is_impossible=is_impossible,
                            )
                        )
            break

        yield doc.model_dump_json()


if __name__ == "__main__":
    local_dir = download_and_unzip()
    service_files = list(collect_target_files())
    annotations = load_annotations(local_dir)
    dicts = list(annotate(service_files, annotations))
    ds = datasets.Dataset.from_pandas(pd.DataFrame(dicts))
    ds = ds.rename_column("0", "document")
    ds.push_to_hub("chenghao/tos_pp_dataset", "cuad")
