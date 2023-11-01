LOG_DIR = "logs"
LOG_FILE = "extraction.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%d/%m/%Y %H:%M:%S"

INPUT_DIR = "data"
INPUT_FILE = "tvpl"

OUTPUT_DIR = "data"
OUTPUT_FILE = "tvpl"

STOPWORDS_DIR = "data"
STOPWORDS_FILE = "vietnamese-stopwords.txt"

SPAN_TYPES = list(["NE", "NP", "AP", "VP", "S"])
POS_REPLACE = dict(
    {"NP": "NOUNPHRASE", "AP": "ADVPHRASE", "VP": "VERBPHARSE", "S": "CLAUSE"}
)

ENTITY_TYPES = list(
    [
        "PLACEHOLDER",
        "PERSON",
        "ORGANIZATION",
        "MISCELLANEOUS",
        "LOCATION",
        "DATE",
        "MONEY",
        "PERCENT",
    ]
)
