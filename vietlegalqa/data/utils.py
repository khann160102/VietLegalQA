import json, pickle
from typing import Dict, Iterator, List, Optional, Union

FIELD = list(["id"])
DOC_FIELD = list(
    [
        FIELD[0],
        "title",
        "summary",
        "context",
    ]
)
QA_FIELD = list(
    [
        FIELD[0],
        "article",
        "question",
        "answer",
        "start",
        "type",
        "is_impossible",
    ]
)


def get_extension(filename: str, type: Optional[str] = None):
    match type:
        case "json":
            return (
                f"{filename.strip()}.json"
                if not filename.strip().endswith(".json")
                else filename.strip()
            )
        case "pickle":
            return (
                f"{filename.strip()}.pkl"
                if not filename.strip().endswith(".pkl")
                else filename.strip()
            )
        case _:
            filename.strip()


class Field:
    id = FIELD[0]


class DocField(Field):
    title = DOC_FIELD[1]
    summary = DOC_FIELD[2]
    context = DOC_FIELD[3]


class QAField(Field):
    article = QA_FIELD[1]
    question = QA_FIELD[2]
    answer = QA_FIELD[3]
    start = QA_FIELD[4]
    type = QA_FIELD[5]
    is_impossible = QA_FIELD[6]


class Entry:
    def __init__(
        self,
        id: Optional[str] = None,
    ) -> None:
        self.id = id

    def __call__(
        self, key: Optional[str] = None
    ) -> Union[str, Dict[str, Union[str, None]], None]:
        try:
            match key:
                case Field.id:
                    return self.id
                case _:
                    return self.to_dict()
        except Exception as e:
            raise e

    def __getitem__(self, key: str) -> Union[str, None]:
        try:
            match key:
                case Field.id:
                    return self.id
                case _:
                    return None
        except Exception as e:
            raise e

    def __str__(self) -> str:
        try:
            return str(self.__call__())
        except Exception as e:
            raise e

    def __repr__(self) -> str:
        try:
            return str(self.__call__())
        except Exception as e:
            raise e

    def to_list(self) -> List[Union[str, None]]:
        try:
            return list([self.id])
        except Exception as e:
            raise e

    def to_dict(self) -> Dict[str, Union[str, None]]:
        try:
            return dict({FIELD[idx]: field for idx, field in enumerate(self.to_list())})
        except Exception as e:
            raise e


class Dataset:
    def __init__(self) -> None:
        self.data: Dict[str, Entry] = dict()

    def __call__(self) -> Dict[str, Entry]:
        try:
            return self.data
        except Exception as e:
            raise e

    def __len__(self) -> int:
        try:
            return len(self.data)
        except Exception as e:
            raise e

    def __getitem__(
        self, key: Union[str, int, slice]
    ) -> Union[Entry, list[Entry], None]:
        try:
            match key:
                case str():
                    return self.data.get(key, Entry())
                case int():
                    return list(self.data.values())[key]
                case slice():
                    return list(self.data.values())[key]
                case _:
                    return None
        except Exception as e:
            raise e

    def __iter__(self) -> Iterator[Entry]:
        try:
            return iter(self.data.values())
        except Exception as e:
            raise e

    def __str__(self) -> str:
        try:
            return "\n\n".join(map(str, self.to_list()))
        except Exception as e:
            raise e

    def __repr__(self) -> str:
        try:
            return str(self.__call__())
        except Exception as e:
            raise e

    def append(self, entry: Entry):
        try:
            self.data[entry.id] = entry
        except Exception as e:
            raise e

    def extend(self, entries: List[Entry]):
        try:
            for entry in entries:
                self.data[entry.id] = entry
        except Exception as e:
            raise e

    def to_list(self) -> List[Dict[str, str | None]]:
        try:
            return list([entry.to_dict() for entry in self.data.values()])
        except Exception as e:
            raise e

    def to_json(
        self, path: str, indent: Optional[int] = 4, ensure_ascii: Optional[bool] = False
    ) -> None:
        try:
            with open(
                file=get_extension(filename=path, type="json"),
                mode="w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    obj=self.to_list(),
                    fp=file,
                    ensure_ascii=ensure_ascii,
                    indent=indent,
                )
        except Exception as e:
            raise e

    def to_pickle(self, path: str) -> None:
        try:
            with open(
                file=get_extension(filename=path, type="pickle"),
                mode="wb",
            ) as file:
                pickle.dump(obj=self, file=file, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise e
