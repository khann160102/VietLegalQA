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


class Field:
    @property
    def id() -> str:
        return FIELD[0]


class DocField(Field):
    @property
    def title() -> str:
        return FIELD[1]

    @property
    def summary() -> str:
        return FIELD[2]

    @property
    def context() -> str:
        return FIELD[3]


class QAField(Field):
    @property
    def article() -> str:
        return FIELD[1]

    @property
    def question() -> str:
        return FIELD[2]

    @property
    def answer() -> str:
        return FIELD[3]

    @property
    def start() -> str:
        return FIELD[4]

    @property
    def type() -> str:
        return FIELD[5]

    @property
    def is_impossible() -> str:
        return FIELD[6]


def get_extension(filename: str, type: Optional[str]) -> str:
    """
    Convert the filename into a formatted one with file extension if the filename does not contain the extension, otherwise return the unchanged filename.

    Args:
        filename (`str`):
            The name of the file.
        type (`str`, default to `None`):
            The desired file extension. It can be either "json" or "pickle". If no `type` is provided, the function will return the filename as is.

    Returns:
        (`str`)
            The filename with the specified extension. If the filename does not already have the specified extension, it will be added.

    Examples:

    Convert a filename that does not have an extension to the `json` type:

    ```py
    >>> filename = get_extension("abc", "json")
    >>> filename
    'abc.json'
    ```

    Convert a filename that already has an extension of `pickle` type:

    ```py
    >>> filename = get_extension("abc.pkl", "pickle")
    >>> filename
    'abc.pkl'
    ```
    """
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
            return filename.strip()


class Entry:
    """Abstract class, presenting an element of an instance of the class `Dataset`."""

    def __init__(
        self,
        id: Optional[str],
    ) -> None:
        """
        Initializes an entry with an optional id parameter.

        Args:
            id (`str`, default to `None`):
                The identifier for the entry.
        """
        self._id = id

    @property
    def id(self):
        """Access to the ID of this entry."""
        return self._id

    @id.setter
    def id(self, value):
        """Set the ID of this entry."""
        self._id = value

    def __call__(
        self, key: Optional[str] = None
    ) -> Union[str, Dict[str, Union[str, None]], None]:
        """
        Returns the value of a specific field if the key matches, otherwise returns a dictionary representation of the object.

        Args:
            key (`str`, default to `None`):
                The `key` parameter is an optional string that represents the property needed to be accessed.

        Returns:
            `str` or `dict`

        Examples:

        Access the ID of the entry:

        ```py
        >>> entry = Entry(id='id_00')
        >>> entry('id')
        'id_00'
        ```

        Call the dictionary of the entry:

        ```py
        >>> entry = Entry(id='id_00')
        >>> entry()
        {'id':'id_00'}
        ```
        """
        try:
            match key:
                case Field.id:
                    return self.id
                case _:
                    return self.to_dict()
        except Exception as e:
            raise e

    def __getitem__(self, key: str) -> Union[str, None]:
        """
        Returns the value of a specific field if the key matches, otherwise returns None.

        Args:
            key (`str`, default to `None`):
                The `key` parameter is an optional string that represents the property needed to be accessed.

        Returns:
            `str` or `NoneType`

        Examples:

        Access the ID of the entry:

        ```py
        >>> entry = Entry(id='id_00')
        >>> entry['id']
        'id_00'
        ```
        """
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
        """
        Converts all the properties of the entry to a list.

        Returns:
          The method `to_list` is returning a list containing the value of `self.id`.
        """
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
