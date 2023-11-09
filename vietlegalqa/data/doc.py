from typing import Dict, List, Optional, Union

from .utils import Entry, Dataset
from .utils import DOC_FIELD as FIELD, DocField as Field


class Article(Entry):
    def __init__(
        self,
        id: Optional[str] = None,
        title: Optional[str] = None,
        summary: Optional[List[str]] = None,
        context: Optional[List[str]] = None,
    ) -> None:
        self.id = id
        self.title = title
        self.summary = summary
        self.context = context

    def __call__(
        self, key: Optional[str] = None
    ) -> Union[str, List[str], Dict[str, Union[str, List[str], None]], None]:
        try:
            match key:
                case Field.id:
                    return self.id
                case Field.title:
                    return self.title
                case Field.summary:
                    return self.summary
                case Field.context:
                    return self.context
                case _:
                    return self.to_dict()
        except Exception as e:
            raise e

    def __getitem__(self, key: str) -> Union[str, List[str], None]:
        try:
            match key:
                case Field.id:
                    return self.id
                case Field.title:
                    return self.title
                case Field.summary:
                    return self.summary
                case Field.context:
                    return self.context
                case _:
                    return None
        except Exception as e:
            raise e

    def to_list(self) -> List[Union[str, List[str], None]]:
        try:
            return list([self.id, self.title, self.summary, self.context])
        except Exception as e:
            raise e

    def to_dict(self) -> Dict[str, Union[str, List[str], None]]:
        try:
            return dict({FIELD[idx]: field for idx, field in enumerate(self.to_list())})
        except Exception as e:
            raise e


class Document(Dataset):
    def __init__(
        self,
        data: Optional[
            Union[List[Dict[str, Union[str, List[str]]]], Dict[str, List[str]]]
        ] = None,
        field: Optional[List[str]] = FIELD,
    ) -> None:
        self.data: Dict[str, Article] = dict()
        try:
            match data:
                case list():
                    for entry in data:
                        self.data[entry[field[0]]] = Article(
                            id=entry[field[0]],
                            title=entry[field[1]],
                            summary=entry[field[2]],
                            context=entry[field[3]],
                        )
                case dict():
                    for idx, id in enumerate(data[field[0]]):
                        self.data[idx] = Article(
                            id=id,
                            title=data[field[1]][idx],
                            summary=data[field[2]][idx],
                            context=data[field[3]][idx],
                        )
                case _:
                    pass
        except Exception as e:
            raise e

    def append(self, entry: Article):
        try:
            self.data[entry.id] = entry
        except Exception as e:
            raise e

    def extend(self, entries: List[Article]):
        try:
            for entry in entries:
                self.data[entry.id] = entry
        except Exception as e:
            raise e
