from typing import Annotated, Optional

from pydantic import BaseModel, Field, model_validator


class Document(BaseModel):
    title: Annotated[str, "The title of the document"]
    text: Annotated[Optional[str], "The full text of the document"] = None
    paragraphs: Annotated[Optional[list[str]], "The paragraphs of the document"] = None
    sentences: Annotated[Optional[list[str]], "The sentences of the document"] = None
    language: Annotated[Optional[str], "The language of the document"] = None

    @model_validator(mode="before")
    def check_at_least_one(cls, data: dict) -> dict:
        if not any([data.get("text"), data.get("paragraphs"), data.get("sentences")]):
            raise ValueError(
                "At least one of text, paragraphs, or sentences must be provided."
            )
        return data


class QA(BaseModel):
    question: Annotated[str, "The question to answer"]
    answer: Annotated[str, "The answer to the question"]
    start: Annotated[
        int | None, "The start index of the answer in the document, inclusive"
    ] = None
    end: Annotated[
        int | None, "The end index of the answer in the document, exclusive"
    ] = None
    is_impossible: Annotated[bool, "Whether the question is impossible to answer"]


class Tag(BaseModel):
    tag: Annotated[str, "The label for a span"]
    start: Annotated[int, "The start index of the answer in the document, inclusive"]
    end: Annotated[int, "The end index of the answer in the document, exclusive"]
    comment: Annotated[str | None, "Additional information for the span"] = None


class Event(BaseModel):
    event_type: Annotated[str, "The event annotated"]
    trigger: Annotated[Tag, "The trigger of the event"]
    arguments: Annotated[list[Tag], "The arguments for the event"]


class Classification(BaseModel):
    level: Annotated[str, "The level of the classification"]
    labels: Annotated[list[str], "The labels of the classification"]
    label_definitions: Annotated[
        list[list[str]] | None, "The definitions of the labels"
    ] = None


class DocumentSequenceClassification(BaseModel):
    document: Document = Field(..., description="The document")
    tags: list[Tag] = Field(..., description="The annotations of the document")


class DocumentEvent(BaseModel):
    document: Document = Field(..., description="The document")
    events: list[Event] = Field(..., description="The events of the document")


class DocumentQA(BaseModel):
    document: Document = Field(..., description="The document to answer the question")
    qas: list[QA] = Field(..., description="The questions and answers to the document")


class DocumentClassification(BaseModel):
    document: Annotated[Document, "The document to classify"]
    classifications: Annotated[
        list[Classification], "The classifications of the document"
    ]


class Service(BaseModel):
    name: Annotated[str, "The name of the service"]
    url: Annotated[str, "The URL of the service agreement"]
    lang: Annotated[str, "The language of the service agreement"]
    sector: Annotated[str, "The sector of the service"]
    hq: Annotated[str, "The headquarters of the service"]
    hq_category: Annotated[str, "The category of the headquarters of the service"]
    is_public: Annotated[str, "Whether the service is public"]
    is_paid: Annotated[str, "Whether the service is paid"]
    date: Annotated[str, "The date of the service agreement"]


class EUConsumerLawAnnotation(BaseModel):
    general_category: Annotated[str, "The general category of the annotation"]
    name: Annotated[str, "The name of the annotation"]
    legal_ground: Annotated[str, "The legal ground of the annotation"]
    code: Annotated[str, "The code of the annotation"]
    score: Annotated[int, "The score of the annotation (-1, 0, or 1)"] = Field(
        ..., ge=-1, le=1
    )
    explanation: Annotated[str, "The explanation of the annotation"]


class DocumentEUConsumerLawAnnotation(BaseModel):
    service: Annotated[Service, "The service metadata"]
    document: Annotated[Document, "The document to annotate"]
    annotations: Annotated[
        list[EUConsumerLawAnnotation], "The annotations of the document"
    ]
