from typing import Dict, List, Optional, Union

from .doc import Article, Document
from .utils import Entry, Dataset
from .utils import QA_FIELD as FIELD, QAField as Field


class QAPair(Entry):
    def __init__(
        self,
        id: Optional[str] = None,
        article: Optional[str] = None,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        start: Optional[int] = None,
        type: Optional[str] = None,
        is_impossible: Optional[bool] = False,
    ) -> None:
        self.id = id
        self.article = article
        self.question = question
        self.answer = answer
        self.start = start
        self.type = type.upper()
        self.is_impossible = is_impossible

    def __call__(
        self, key: Optional[str] = None
    ) -> Union[str, int, bool, Dict[str, Union[int, str, bool, None]], None]:
        try:
            match key:
                case Field.id:
                    return self.id
                case Field.article:
                    return self.article
                case Field.question:
                    return self.question
                case Field.answer:
                    return self.answer
                case Field.start:
                    return self.start
                case Field.type:
                    return self.type
                case Field.is_impossible:
                    return self.is_impossible
                case _:
                    return self.to_dict()
        except Exception as e:
            raise e

    def __getitem__(self, key: Optional[str] = None) -> Union[str, int, bool, None]:
        try:
            match key:
                case Field.id:
                    return self.id
                case Field.article:
                    return self.article
                case Field.question:
                    return self.question
                case Field.answer:
                    return self.answer
                case Field.start:
                    return self.start
                case Field.type:
                    return self.type
                case Field.is_impossible:
                    return self.is_impossible
                case _:
                    return None
        except Exception as e:
            raise e

    def __eq__(self, __value: object) -> bool:
        try:
            if isinstance(__value, QAPair):
                return (
                    True
                    if self.article == __value.article
                    and self.question == __value.question
                    and self.answer == __value.answer
                    and self.start == __value.start
                    else False
                )
            else:
                return False
        except Exception as e:
            raise e

    def __ne__(self, __value: object) -> bool:
        try:
            if isinstance(__value, QAPair):
                return (
                    True
                    if self.article != __value.article
                    and self.question != __value.question
                    and self.answer != __value.answer
                    and self.start != __value.start
                    else False
                )
            else:
                return False
        except Exception as e:
            raise e

    def __lt__(self, __value: object) -> bool:
        try:
            if isinstance(__value, QAPair):
                if self.article < __value.article:
                    return True
                elif self.question < __value.question:
                    return True
                elif self.answer < __value.answer:
                    return True
                elif self.start < __value.start:
                    return True
                else:
                    return False
            else:
                return False
        except Exception as e:
            raise e

    def __gt__(self, __value: object) -> bool:
        try:
            if isinstance(__value, QAPair):
                if self.article > __value.article:
                    return True
                elif self.question > __value.question:
                    return True
                elif self.answer > __value.answer:
                    return True
                elif self.start > __value.start:
                    return True
                else:
                    return False
            else:
                return False
        except Exception as e:
            raise e

    def __le__(self, __value: object) -> bool:
        try:
            if isinstance(__value, QAPair):
                if self.article <= __value.article:
                    return True
                elif self.question <= __value.question:
                    return True
                elif self.answer <= __value.answer:
                    return True
                elif self.start <= __value.start:
                    return True
                else:
                    return False
            else:
                return False
        except Exception as e:
            raise e

    def __ge__(self, __value: object) -> bool:
        try:
            if isinstance(__value, QAPair):
                if self.article >= __value.article:
                    return True
                elif self.question >= __value.question:
                    return True
                elif self.answer >= __value.answer:
                    return True
                elif self.start >= __value.start:
                    return True
                else:
                    return False
            else:
                return False
        except Exception as e:
            raise e

    def __cmp__(self, __value: object) -> int:
        try:
            if isinstance(__value, QAPair):
                if self == __value:
                    return 0
                elif self > __value:
                    return 1
                else:
                    return -1
        except Exception as e:
            raise e

    def to_list(self) -> List[Union[str, int, bool, None]]:
        try:
            return list(
                [
                    self.id,
                    self.article,
                    self.question,
                    self.answer,
                    self.start,
                    self.type,
                    self.is_impossible,
                ]
            )
        except Exception as e:
            raise e

    def to_dict(self) -> Dict[str, Union[str, int, bool, None]]:
        try:
            return dict({FIELD[idx]: field for idx, field in enumerate(self.to_list())})
        except Exception as e:
            raise e

    def get_article(self, document: Document) -> Article:
        try:
            return document[self.article]
        except Exception as e:
            raise (e)


class QADataset(Dataset):
    def __init__(
        self,
        data: Optional[
            Union[
                List[Dict[str, Union[str, int, bool]]], Dict[str, Union[str, int, bool]]
            ]
        ] = None,
        field: Optional[List[str]] = FIELD,
    ) -> None:
        self.data: Dict[str, QAPair] = dict()
        try:
            match data:
                case list():
                    for entry in data:
                        self.data[entry[field[0]]] = QAPair(
                            id=entry[field[0]],
                            article=entry[field[1]],
                            question=entry[field[2]],
                            answer=entry[field[3]],
                            start=entry[field[4]],
                            type=entry[field[5]],
                            is_impossible=entry[field[6]],
                        )
                case dict():
                    for idx, id in enumerate(data[field[0]]):
                        self.data[id] = QAPair(
                            id=id,
                            article=data[field[1]][idx],
                            question=data[field[2]][idx],
                            answer=data[field[3]][idx],
                            start=data[field[4]][idx],
                            type=data[field[5]][idx],
                            is_impossible=data[field[6]][idx],
                        )
                case _:
                    pass
        except Exception as e:
            raise e

    def append(self, entry: QAPair):
        try:
            self.data[entry.id] = entry
        except Exception as e:
            raise e

    def extend(self, entries: List[QAPair]):
        try:
            for entry in entries:
                self.data[entry.id] = entry
        except Exception as e:
            raise e

    def get_article(self, id: str, document: Document) -> Article:
        try:
            return document[self.data[id]]
        except Exception as e:
            raise (e)
