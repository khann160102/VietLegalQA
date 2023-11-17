import argparse
from vietlegalqa import load_document_hf, QAConstruct
from stanza import Pipeline
from torch.cuda import is_available, device_count


FIELD = ["url", "title", "summary", "document"]
DOC_HF = "vietlegalqa/tvpl_summary_kha"
STOPWORDS_DIR = "./data/vietnamese-stopwords.txt"
PREFIX = "tvpl"


def check_args(args) -> tuple[bool, str]:
    if args.use_gpu is True:
        if not is_available:
            setattr(args, "use_gpu", False)

    if args.device >= device_count() or args.device < 0:
        setattr(args, "device", 0)

    return True


def main(args):
    doc = load_document_hf(path=args.doc, split="train", field=FIELD)

    with open(
        args.stopwords_dir,
        "r",
        encoding="utf-8",
    ) as stopwords_file:
        STOPWORDS = stopwords_file.read().splitlines()

    PARSER = Pipeline(
        lang=args.lang,
        processors="tokenize, pos, ner, constituency",
        use_gpu=args.use_gpu,
        device=args.device,
        verbose=args.verbose,
        allow_unknown_language=True,
        tokenize_pretokenized=True,
        tokenize_no_ssplit=True,
    )
    POS = Pipeline(
        lang=args.lang,
        processors="tokenize, pos, lemma",
        use_gpu=args.use_gpu,
        device=args.device,
        verbose=args.verbose,
        allow_unknown_language=True,
        tokenize_pretokenized=True,
        tokenize_no_ssplit=True,
    )

    constructor = QAConstruct(stopwords=STOPWORDS, parser=PARSER, pos=POS)
    qa = constructor(document=doc, id_prefix=args.id_prefix)
    qa.to_pickle("./data/tvpl_contruct.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default=DOC_HF, type=str)
    parser.add_argument("--stopwords_dir", default=STOPWORDS_DIR, type=str)
    parser.add_argument("--id_prefix", default=PREFIX, type=str)
    parser.add_argument("--output_file", default=f"{PREFIX}_construct.py", type=str)
    parser.add_argument("--lang", default="vi", type=str)
    parser.add_argument("--use_gpu", default=True, type=bool)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--verbose", default=0, type=bool)
    args = parser.parse_args()

    if check_args(args):
        main(args)
