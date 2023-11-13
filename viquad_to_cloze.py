import evaluate, argparse

from datasets import load_dataset, concatenate_datasets
from stanza import Pipeline
from stanza.models.constituency.parse_tree import Tree
from statistics import mode
from torch.cuda import is_available, device_count
from tqdm import tqdm
from typing import Any, Dict, List
from underthesea import word_tokenize, sent_tokenize

from utils import *


class Processor:
    def __init__(
        self, args, name: str, splits: List, renames: Dict[str, str], remove: List[str]
    ) -> None:
        dataset = load_dataset(name)

        self.data = (
            concatenate_datasets([dataset[split] for split in splits])
            .rename_columns(renames)
            .remove_columns(remove)
        )
        print(f"Loaded {len(self.data)} data!")
        self.keys: List = self.data.column_names
        for key in NEW_KEYS:
            self.keys.append(key)
        self.tokenizer = Pipeline(
            lang=args.lang,
            processors="tokenize",
            use_gpu=args.use_gpu,
            device=args.device,
            verbose=args.verbose,
            allow_unknown_language=True,
            tokenize_no_ssplit=True,
        )
        self.ner = Pipeline(
            lang=args.lang,
            processors="tokenize, ner",
            use_gpu=args.use_gpu,
            device=args.device,
            verbose=args.verbose,
            allow_unknown_language=True,
            tokenize_pretokenized=True,
            tokenize_no_ssplit=True,
        )
        self.parser = Pipeline(
            lang=args.lang,
            processors="tokenize, pos, constituency",
            use_gpu=args.use_gpu,
            device=args.device,
            verbose=args.verbose,
            allow_unknown_language=True,
        )
        self.rouge = evaluate.load("rouge")

    def get_ner(self, text: str, threshold=THRESHOLD):
        ner = self.ner(ner_tokenize(text)).entities
        score = self.rouge.compute(
            predictions=[" ".join([ent.text for ent in ner])],
            references=[text],
        )
        return mode([ent.type for ent in ner]) if score["rougeL"] >= threshold else None

    def get_labels(self, node: Tree):
        if node.is_leaf():
            return None
        if node.label in POS_TAGS:
            return node.label
        else:
            for child in node.children:
                return self.get_labels(child)

    def get_pos(self, text: str):
        doc = self.parser(" ".join(word_tokenize(text)))
        pos = []
        for sent in doc.sentences:
            labels = self.get_labels(sent.constituency)
            if labels is not None:
                pos.append(labels)
        return (
            "MISCELLANEOUS"
            if len(pos) == 0
            else POS_REPLACE[mode([tag for tag in pos if tag is not None])]
        )

    def get_type(self, answer: str):
        ner = self.get_ner(answer)
        return ner if ner is not None else self.get_pos(answer)

    def get_type_batch(self, batch):
        return [self.get_type(entry) for entry in batch]

    def get_cloze(self, entry):
        end_char = 0
        for sent in self.tokenizer(
            "\n\n".join(sent_tokenize(entry["context"]))
        ).sentences:
            end_char += sent.tokens[-1].end_char
            if entry["start"] <= end_char and entry["answer"] in sent.text:
                return sent.text.replace(entry["answer"], entry["type"])
        return entry["context"].replace(entry["answer"], entry["type"])

    def get_cloze_batch(self, batch):
        entries = []
        for i in range(len(batch[self.keys[0]])):
            entry = {key: batch[key][i] for key in self.keys}
            entries.append(entry)

        return [" ".join(word_tokenize(self.get_cloze(entry))) for entry in entries]

    def __call__(self) -> Any:
        print("Start process answer type")
        self.data = self.data.map(
            lambda batch: {"type": self.get_type_batch(batch["answer"])},
            batched=True,
            batch_size=int(len(self.data) / 10),
        )
        print("Start process cloze answer")
        self.data = self.data.map(
            lambda batch: {"cloze_question": self.get_cloze_batch(batch)},
            batched=True,
            batch_size=int(len(self.data) / 10),
        )

    def push_to_hub(self, name, token):
        self.data.push_to_hub(name, token=token)


def check_args(args) -> tuple[bool, str]:
    msg = []
    if args.use_gpu is True:
        if not is_available:
            setattr(args, "use_gpu", False)

    if args.device >= device_count() or args.device < 0:
        setattr(args, "device", 0)

    return True, msg


def main(args):
    prossesor = Processor(
        args, name=NAME, splits=SPLITS, renames=RENAMES, remove=REMOVES
    )
    prossesor()
    prossesor.push_to_hub(args.push_to_hub, args.token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push_to_hub", type=str)
    parser.add_argument("--token", type=str)
    parser.add_argument("--threshold", default=THRESHOLD, type=float)
    parser.add_argument("--lang", default="vi", type=str)
    parser.add_argument("--use_gpu", default=True, type=bool)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--verbose", default=0, type=bool)
    args = parser.parse_args()

    if check_args(args)[0]:
        main(args)
