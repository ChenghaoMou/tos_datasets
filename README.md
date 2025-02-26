# A collection of Terms of Service or Privacy Policy datasets

## Annotated datasets

### CUAD

Specifically, the 28 service agreements from [CUAD](https://www.atticusprojectai.org/cuad), which are licensed under CC BY 4.0 (subset: `cuad`).

<details>
<summary>Code</summary>

```python
import datasets
from tos_datasets.proto import DocumentQA

ds = datasets.load_dataset("chenghao/tos_pp_dataset", "cuad")

print(DocumentQA.model_validate_json(ds["document"][0]))
```

</details>

### 100 ToS

From [Annotated 100 ToS](https://data.mendeley.com/datasets/dtbj87j937/3), CC BY 4.0 (subset: `100_tos`).

<details>
<summary>Code</summary>

```python
import datasets
from tos_datasets.proto import DocumentEUConsumerLawAnnotation

ds = datasets.load_dataset("chenghao/tos_pp_dataset", "100_tos")

print(DocumentEUConsumerLawAnnotation.model_validate_json(ds["document"][0]))
```

</details>

### Multilingual Unfair Clause

From [CLAUDETTE](http://claudette.eui.eu/corpora/index.html)/[Multilingual Unfair Clause](https://github.com/nlp-unibo/Multilingual-Unfair-Clause-Detection), CC BY 4.0 (subset: `multilingual_unfair_clause`).

It was built from [CLAUDETTE](http://claudette.eui.eu/corpora/index.html)/[25 Terms of Service in English, Italian, German, and Polish (100 documents in total) from A Corpus for Multilingual Analysis of Online Terms of Service](http://claudette.eui.eu/corpus_multilingual_NLLP2021.zip).

<details>
<summary>Code</summary>

```python
import datasets
from tos_datasets.proto import DocumentClassification

ds = datasets.load_dataset("chenghao/tos_pp_dataset", "multilingual_unfair_clause")

print(DocumentClassification.model_validate_json(ds["document"][0]))
```

</details>

### Memnet ToS

From [100 Terms of Service in English from Detecting and explaining unfairness in consumer contracts through memory networks](https://github.com/federicoruggeri/Memnet_ToS), MIT (subset: `memnet_tos`).

<details>

<summary>Code</summary>

```python
import datasets
from tos_datasets.proto import DocumentClassification

ds = datasets.load_dataset("chenghao/tos_pp_dataset", "memnet_tos")

print(DocumentClassification.model_validate_json(ds["document"][0]))
```

</details>

### 142 ToS

From [142 Terms of Service in English divided according to market sector from Assessing the Cross-Market Generalization Capability of the CLAUDETTE System](http://claudette.eui.eu/corpus_142_ToS.zip), Unknown (subset: `142_tos`). This should also includes [50 Terms of Service in English from "CLAUDETTE: an Automated Detector of Potentially Unfair Clauses in Online Terms of Service"](http://claudette.eui.eu/ToS.zip).

<details>
<summary>Code</summary>

```python
import datasets
from tos_datasets.proto import DocumentClassification

ds = datasets.load_dataset("chenghao/tos_pp_dataset", "142_tos")

print(DocumentClassification.model_validate_json(ds["document"][0]))
```

</details>

### 10 ToS/PP

From [5 Terms of Service and 5 Privacy Policies in English and German (10 documents in total) from Cross-lingual Annotation Projection in Legal Texts](https://bitbucket.org/a-galaxy/cross-lingual-annotation-projection-in-legal-texts), GNU GPL 3.0 (subset: `10_tos`)

<details>
<summary>Code</summary>

```python
import datasets
from tos_datasets.proto import DocumentClassification

ds = datasets.load_dataset("chenghao/tos_pp_dataset", "10_tos")

print(DocumentClassification.model_validate_json(ds["document"][0]))
```

</details>

### PolicyQA

> [!IMPORTANT]
> This dataset seems to have some annotation issues where __unanswerable__ questions are still answered with SQuAD-v1 format instead of the v2 format.

From [PolicyQA](https://github.com/wasiahmad/PolicyQA), MIT (subset: `privacy_glue/policy_qa`).

<details>
<summary>Code</summary>

```python
import datasets
from tos_datasets.proto import DocumentQA

ds = datasets.load_dataset("chenghao/tos_pp_dataset", "privacy_glue/policy_qa")

print(DocumentQA.model_validate_json(ds["train"]["document"][0]))
```

</details>

### PolicyIE

From [PolicyIE](https://github.com/wasiahmad/PolicyIE), MIT (subset: `privacy_glue/policy_ie`).

<details>
<summary>Code</summary>

```python
import datasets
from tos_datasets.proto import DocumentSequenceClassification, DocumentEvent

ds = datasets.load_dataset("chenghao/tos_pp_dataset", "privacy_glue/policy_ie")

print(DocumentSequenceClassification.model_validate_json(ds["train"]["type_i"][0]))
print(DocumentEvent.model_validate_json(ds["train"]["type_ii"][0]))
```

</details>

### Policy Detection

From [policy-detection-data](<https://github.com/infsys-lab/policy-detection-data>, GPL 3.0 (subset: `privacy_glue/policy_detection`).

<details>
<summary>Code</summary>

```python
import datasets
from tos_datasets.proto import DocumentClassification

ds = datasets.load_dataset("chenghao/tos_pp_dataset", "privacy_glue/policy_detection")

print(DocumentClassification.model_validate_json(ds["train"]["document"][0]))
```

</details>

### Polisis

From [Polisis](https://github.com/SmartDataAnalytics/Polisis_Benchmark), Unknown (subset: `privacy_glue/polisis`).

<details>
<summary>Code</summary>

```python
import datasets
from tos_datasets.proto import DocumentClassification

ds = datasets.load_dataset("chenghao/tos_pp_dataset", "privacy_glue/polisis")

print(DocumentClassification.model_validate_json(ds["test"]["document"][0]))
```

</details>

### PrivacyQA

From [PrivacyQA](https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP), MIT (subset: `privacy_qa`).

<details>
<summary>Code</summary>

```python
import datasets
from tos_datasets.proto import DocumentClassification

ds = datasets.load_dataset("chenghao/tos_pp_dataset", "privacy_glue/privacy_qa")

print(DocumentClassification.model_validate_json(ds["test"]["document"][0]))
```

</details>

### Piextract

From [Piextract](https://github.com/um-rtcl/piextract_dataset), Unknown (subset: `privacy_glue/piextract`).

<details>
<summary>Code</summary>

```python
import datasets
from tos_datasets.proto import DocumentSequenceClassification

ds = datasets.load_dataset("chenghao/tos_pp_dataset", "privacy_glue/piextract")

print(DocumentSequenceClassification.model_validate_json(ds["train"]["document"][0]))
```

</details>

## WIP

- <del>[Annotated Italian TOS sentences](https://github.com/i3-fbk/LLM-PE_Terms_and_Conditions_Contracts), Apache 2.0</del> Only sentence level annotations, missing original full text
- <del>[Huggingface](https://huggingface.co/datasets/CodeHima/TOS_Dataset), MIT</del> Only sentence level annotations, missing original full text
- [ ] [ToSDR API](https://developers.tosdr.org/dev/get-service-v2), Unknown
