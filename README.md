# A collection of Terms of Service or Privacy Policy datasets

## Annotated datasets

### CUAD

Specifically, the 28 service agreements from [CUAD](https://www.atticusprojectai.org/cuad), which are licensed under CC BY 4.0 (subset: `cuad`).

```python
import datasets
from tos_datasets.proto import DocumentQA

ds = datasets.load_dataset("chenghao/tos_pp_dataset", "cuad")

print(DocumentQA.model_validate_json(ds["document"][0]))
```

### 100 ToS

From [Annotated 100 ToS](https://data.mendeley.com/datasets/dtbj87j937/3), CC BY 4.0 (subset: `100_tos`).

```python
import datasets
from tos_datasets.proto import DocumentEUConsumerLawAnnotation

ds = datasets.load_dataset("chenghao/tos_pp_dataset", "100_tos")

print(DocumentEUConsumerLawAnnotation.model_validate_json(ds["document"][0]))
```

### Multilingual Unfair Clause

From [CLAUDETTE](http://claudette.eui.eu/corpora/index.html)/[Multilingual Unfair Clause](https://github.com/nlp-unibo/Multilingual-Unfair-Clause-Detection), CC BY 4.0 (subset: `multilingual_unfair_clause`).

It was built from [CLAUDETTE](http://claudette.eui.eu/corpora/index.html)/[25 Terms of Service in English, Italian, German, and Polish (100 documents in total) from A Corpus for Multilingual Analysis of Online Terms of Service](http://claudette.eui.eu/corpus_multilingual_NLLP2021.zip).

```python
import datasets
from tos_datasets.proto import DocumentClassification

ds = datasets.load_dataset("chenghao/tos_pp_dataset", "multilingual_unfair_clause")

print(DocumentClassification.model_validate_json(ds["document"][0]))
```

### Memnet ToS

From [100 Terms of Service in English from Detecting and explaining unfairness in consumer contracts through memory networks](https://github.com/federicoruggeri/Memnet_ToS), MIT (subset: `memnet_tos`).

```python
import datasets
from tos_datasets.proto import DocumentClassification

ds = datasets.load_dataset("chenghao/tos_pp_dataset", "memnet_tos")

print(DocumentClassification.model_validate_json(ds["document"][0]))
```

## WIP

- [ ] [Annotated Italian TOS sentences](https://github.com/i3-fbk/LLM-PE_Terms_and_Conditions_Contracts), Apache 2.0
- [ ] [Huggingface](https://huggingface.co/datasets/CodeHima/TOS_Dataset), MIT
- [CLAUDETTE](http://claudette.eui.eu/corpora/index.html):
  - [ ] [142 Terms of Service in English divided according to market sector from Assessing the Cross-Market Generalization Capability of the CLAUDETTE System](http://claudette.eui.eu/corpus_142_ToS.zip), Unknown
  - [ ] [5 Terms of Service and 5 Privacy Policies in English and German (10 documents in total) from Cross-lingual Annotation Projection in Legal Texts](https://bitbucket.org/a-galaxy/cross-lingual-annotation-projection-in-legal-texts), GNU GPL 3.0
  - [ ] [50 Terms of Service in English from "CLAUDETTE: an Automated Detector of Potentially Unfair Clauses in Online Terms of Service"](http://claudette.eui.eu/ToS.zip), Unknown
  - Available demo from [CLAUDETTE](http://claudette.eui.eu/demo/)
- [ ] [PrivacyGLUE](https://github.com/infsys-lab/privacy-glue), GPL 3.0
- [ ] [ToSDR API](https://developers.tosdr.org/dev/get-service-v2), Unknown
