import json, pickle
from typing import Any, List, Optional, Tuple, Union
from datasets import load_dataset
from torch import NoneType

from .doc import Document
from .qa import QADataset
from .utils import DOC_FIELD, get_extension


def load_document_hf(
    path: str,
    split: Optional[str] = "train",
    field: Optional[List[str]] = DOC_FIELD,
    select: Optional[Union[int, Tuple[int, int], Tuple[int, int, int]]] = None,
) -> Document:
    try:
        return Document(data=load_dataset(path)[split].to_list(), field=field)
        match range:
            case NoneType():
                return Document(data=load_dataset(path)[split].to_list(), field=field)
            case int():
                return Document(
                    data=load_dataset(path)[split].select(range(select)).to_list(),
                    field=field,
                )
            case tuple():
                return (
                    Document(
                        data=load_dataset(path, split=split)
                        .select(range(select[0], select[1]))
                        .to_list(),
                        field=field,
                    )
                    if len(select) == 2
                    else Document(
                        data=load_dataset(path, split=split)
                        .select(range(select[0], select[1], select[2]))
                        .to_list(),
                        field=field,
                    )
                )
    except Exception as e:
        raise e


def load_document(
    path: str, type: Optional[str] = None, field: Optional[List[str]] = DOC_FIELD
) -> Union[Document, Any, None]:
    try:
        match type:
            case "json":
                with open(
                    file=get_extension(filename=path, type="json"),
                    mode="r",
                    encoding="utf-8",
                ) as file:
                    return Document(data=json.load(fp=file), field=field)
            case "pickle":
                with open(
                    file=get_extension(filename=path, type="pickle"),
                    mode="rb",
                ) as file:
                    return pickle.load(file=file)
    except Exception as e:
        raise e


def load_qa_hf(
    path: str,
    split: Optional[str] = "train",
    field: Optional[List[str]] = DOC_FIELD,
    select: Optional[Union[int, Tuple[int, int], Tuple[int, int, int]]] = None,
) -> QADataset:
    try:
        match range:
            case None:
                return QADataset(data=load_dataset(path)[split].to_list(), field=field)
            case int():
                return QADataset(
                    data=load_dataset(path)[split].select(range(select)).to_list(),
                    field=field,
                )
            case tuple():
                return (
                    QADataset(
                        data=load_dataset(path, split=split)
                        .select(range(select[0], select[1]))
                        .to_list(),
                        field=field,
                    )
                    if len(select) == 2
                    else QADataset(
                        data=load_dataset(path, split=split)
                        .select(range(select[0], select[1], select[2]))
                        .to_list(),
                        field=field,
                    )
                )
    except Exception as e:
        raise e


def load_qa(
    path: str, type: Optional[str] = None, field: Optional[List[str]] = DOC_FIELD
) -> Union[QADataset, Any, None]:
    try:
        match type:
            case "json":
                with open(
                    file=get_extension(filename=path, type="json"),
                    mode="r",
                    encoding="utf-8",
                ) as file:
                    return QADataset(data=json.load(fp=file), field=field)
            case "pickle":
                with open(
                    file=get_extension(filename=path, type="pickle"),
                    mode="rb",
                ) as file:
                    return pickle.load(file=file)
    except Exception as e:
        raise e
