from typing import Dict, List
from underthesea import sent_tokenize, word_tokenize

NAME: str = "phatjk/viquad"
SPLITS: List = list(
    [
        "train",
        "test",
    ]
)
RENAMES: Dict[str, str] = dict(
    {
        "Id": "id",
        "ans_start": "start",
        "text": "answer",
    }
)
REMOVES: List[str] = list(
    ["__index_level_0__"],
)

NEW_KEYS: List[str] = list(["type"])
THRESHOLD: float = 0.8
POS_TAGS: List[str] = list(
    [
        "NUM",
        "NP",
        "AP",
        "VP",
        "S",
    ]
)
POS_REPLACE: Dict[str, str] = dict(
    {
        "NUM": "NUMBER",
        "NP": "NOUNPHRASE",
        "AP": "ADVPHRASE",
        "VP": "VERBPHARSE",
        "S": "CLAUSE",
    }
)


def ner_tokenize(text):
    return "\n\n".join(["\n".join(word_tokenize(sent)) for sent in sent_tokenize(text)])
