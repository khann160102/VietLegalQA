LOG_FILE = "answer_extraction.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%d/%m/%Y %H:%M:%S"

DATA_DIR = "data"
INPUT_FILE = "tvpl"
STOPWORDS_FILE = "vietnamese-stopwords.txt"

SPAN_TYPES = ["NE", "NP", "AP", "VP", "S"]
ENTITY_TYPES = [
    "PLACEHOLDER",
    "PERSON",
    "ORGANIZATION",
    "MISCELLANEOUS",
    "LOCATION",
    "DATE",
    "MONEY",
    "PERCENT",
]
