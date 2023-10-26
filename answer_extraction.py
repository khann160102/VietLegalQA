import argparse, json, logging, os
from math import log
import stanza

from stanza.models.common.doc import Document
from stanza.models.constituency.parse_tree import Tree
from tqdm import tqdm

from utils import (
    DATA_DIR,
    INPUT_FILE,
    STOPWORDS_FILE,
    SPAN_TYPES,
    LOG_FILE,
    LOG_FORMAT,
    DATE_FORMAT,
)

logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    level=logging.INFO,
    encoding="utf-8",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)


class AnswerExtractor:
    def __init__(self, args) -> None:
        self.args = args

        try:
            with open(
                os.path.join(self.args.input_dir, self.args.input_file),
                "r",
                encoding="utf-8",
            ) as input_file:
                self.data = json.load(input_file)
        except Exception as e:
            logging.error(e)
        logging.info(f"Data imported! Entry: {len(self.data)}")

        try:
            with open(
                os.path.join(self.args.stopwords_dir, self.args.stopwords_file),
                "r",
                encoding="utf-8",
            ) as stopwords_file:
                self.STOPWORDS = stopwords_file.read().splitlines()
        except Exception as e:
            logging.error(e)
        logging.info("Stop words library import!")

        try:
            self.PARSER = stanza.Pipeline(
                lang=self.args.lang,
                processors="tokenize, pos, ner, constituency",
                use_gpu=self.args.use_gpu,
                device=self.args.device,
                verbose=False,
            )
        except Exception as e:
            logging.error(e)
        logging.info("Constituency Parser loaded!")

        try:
            self.POS = stanza.Pipeline(
                lang=self.args.lang,
                processors="tokenize, pos, lemma",
                use_gpu=self.args.use_gpu,
                device=self.args.device,
                verbose=False,
            )
        except Exception as e:
            logging.error(e)
        logging.info("POS Tagger loaded!")

    def is_stop(self, word: str):
        return True if word in self.STOPWORDS else False

    def get_answer_start(self, answer: str, question: str, context: list[str]) -> int:
        q_tokens = []
        for word in self.POS(question).sentences[0].words:
            if not self.is_stop(word):
                q_tokens.append(word.lemma)

        answer_rank = []
        for ctx in context:
            if ctx.find(answer) != -1:
                score = 0
                for words in self.POS(ctx).sentences[0].words:
                    if not self.is_stop(words) and words.lemma in q_tokens:
                        score += 1
                answer_rank.append({"score": score, "context": ctx})

        answer_start = -1

        if len(answer_rank) != 0:
            answer_rank = sorted(answer_rank, key=lambda x: x["score"])
            refined_answer = answer_rank[0]["context"]
            answer_start = " ".join(context).find(refined_answer) + refined_answer.find(
                answer
            )

        return answer_start

    def extract_s_clauses(self, node: Tree, threshold: int == 3) -> list:
        if node.is_leaf():
            return []
        clauses = []

        for leaf in node.children:
            clauses += self.extract_s_clauses(leaf, threshold=threshold)

        if node.label == "S" and len(node.leaf_labels()) > threshold:
            clauses.append(" ".join(node.leaf_labels()).strip())

        return clauses

    def extract_comma_clauses(self, tree: Tree) -> list:
        clauses = []
        comma_clauses = " ".join(tree.leaf_labels()).split(",")

        if len(comma_clauses) > 1:
            idx = 0
            for clause in comma_clauses:
                idx += 1

                while len(clause) < 10 and idx < len(comma_clauses):
                    clause = ", ".join([clause.strip(), comma_clauses[idx].strip()])
                    idx += 1

                clauses.append(clause)
        return clauses

    def extract_clauses(self, nlp: Document) -> list:
        clauses = []

        for sent in nlp.sentences:
            try:
                clauses += self.extract_s_clauses(sent.constituency, 3)
            except Exception as e:
                logging.error(e)

            try:
                clauses += self.extract_comma_clauses(sent.constituency)
            except Exception as e:
                logging.error(e)

        clauses = sorted(clauses, key=lambda clause: len(clause))
        return clauses

    def extract_pos(self, node: Tree, pos_tag: str) -> list:
        if node.is_leaf():
            return []

        spans = []
        for child in node.children:
            spans += self.extract_pos(child, pos_tag)

        if pos_tag == node.label.upper():
            spans.append(" ".join(node.leaf_labels()))

        return spans

    def extract_pos_answers(self, nlp: Document, pos_tag: str) -> list:
        spans = []
        try:
            for sent in nlp.sentences:
                spans += self.extract_pos(sent.constituency, pos_tag)
        except Exception as e:
            return []

        return spans

    def get_summary(self, doc: str) -> tuple:
        # doc = " ".join(word_tokenize(doc))
        nlp = self.PARSER(doc)
        doc = ""
        for sent in nlp.sentences:
            doc += " ".join(sent.constituency.leaf_labels()) + " "
        doc = doc.strip()

        return doc, nlp

    def to_qa(
        self,
        id: str,
        is_impossible: bool,
        question: str,
        answer: str,
        answer_type: str,
        answer_start: int,
        answer_id,
    ):
        return {
            "id": id,
            "is_impossible": is_impossible,
            "question": question,
            "answers": [
                {
                    "text": answer,
                    "type": answer_type,
                    "start": answer_start,
                    "id": answer_id,
                }
            ],
        }

    def extract_answer_pos(self, span_type: str) -> list:
        cloze = []
        count = 0

        for item in tqdm(self.data, desc="Answer Extraction"):
            summaries = []
            qas = []

            for idx, summary in enumerate(item["summary"]):
                summary, summary_doc = self.get_summary(summary)
                summaries.append(summary)

                try:
                    spans = self.extract_pos_answers(summary_doc, span_type)
                    if len(spans) == 0:
                        continue
                except Exception as e:
                    logging.error(e)

                try:
                    clauses = self.extract_clauses(summary_doc)
                    if len(clauses) == 0:
                        continue
                except Exception as e:
                    logging.error(e)

                for answer in spans:
                    if len(answer) == 0:
                        continue

                    question = None
                    for clause in clauses:
                        if len(answer) != 0 and clause.find(answer) != -1:
                            question = clause.replace(answer, "PLACEHOLDER", 1)
                            break
                    if not question:
                        continue

                    try:
                        answer_start = self.get_answer_start(
                            answer, question, item["document"]
                        )
                    except Exception as e:
                        logging.error(e)

                    if answer_start == -1:
                        continue

                    qas.append(
                        self.to_qa(
                            id=f'{item["url"]}_{count}',
                            is_impossible=False,
                            question=question,
                            answer=answer,
                            answer_type=span_type,
                            answer_start=answer_start,
                            answer_id=idx,
                        )
                    )
                    count += 1

            cloze.append(
                {
                    "title": item["title"],
                    "summary": summaries,
                    "context": " ".join(item["document"]),
                    "QA": qas,
                }
            )

        logging.info(f"Answers extracted: {count}")
        return cloze

    def extract_answer_ne(self):
        cloze = []
        count = 0

        for item in tqdm(self.data, desc="Answers Extraction"):
            summaries = []
            qas = []

            for idx, summary in enumerate(item["summary"]):
                summary, summary_doc = self.get_summary(summary)
                summaries.append(summary)

                try:
                    clauses = self.extract_clauses(summary_doc)
                    if len(clauses) == 0:
                        continue
                except Exception as e:
                    logging.error(e)

                for answer in summary_doc.ents:
                    if len(answer.text) == 0:
                        continue

                    question = None
                    for clause in clauses:
                        if clause.find(answer.text) != -1:
                            question = clause.replace(answer.text, answer.text, 1)
                            break

                    if not question:
                        for sent in summary_doc.sentences:
                            if answer.end_char <= sent.tokens[-1].end_char:
                                question = " ".join(
                                    sent.constituency.leaf_labels()
                                ).replace(
                                    answer.text,
                                    answer.type,
                                    1,
                                )
                                break
                    if not question:
                        continue

                    answer_start = self.get_answer_start(
                        answer.text, question, item["document"]
                    )
                    if answer_start == -1:
                        continue

                    qas.append(
                        self.to_qa(
                            id=f'{item["url"]}_{count}',
                            is_impossible=False,
                            question=question,
                            answer=answer.text,
                            answer_type=answer.type,
                            answer_start=answer_start,
                            answer_id=idx,
                        )
                    )
                    count += 1

            cloze.append(
                {
                    "title": item["title"],
                    "summary": summaries,
                    "context": " ".join(item["document"]),
                    "QA": qas,
                }
            )

        logging.info(f"Answers extracted: {count}")
        return cloze

    def to_json(self, data: list[tuple], span_type: str):
        try:
            filename = os.path.join(
                self.args.output_dir,
                f"{self.args.input_file}_answers_extract_{span_type}.json",
            )
            json.dump(
                data,
                open(
                    filename,
                    "w",
                    encoding="utf-8",
                ),
                indent=4,
                ensure_ascii=False,
            )
            logging.info(f"Saved results to {filename}")

        except Exception as e:
            logging.error(e)

    def extract_answer(self, span_type: str):
        try:
            logging.info(f"Extracting: {span_type}")
            cloze = (
                self.extract_answer_ne()
                if span_type == "NE"
                else self.extract_answer_pos(span_type=span_type)
            )

            self.to_json(data=cloze, span_type=span_type)

        except Exception as e:
            logging.error(e)

    def extract(self):
        try:
            if self.args.span_type.upper() == "ALL":
                logging.info("Extracting all span types...")
                for type in SPAN_TYPES:
                    self.extract_answer(span_type=type)
            elif self.span_type in SPAN_TYPES:
                logging.info(f"Extracting {self.span_type} span types...")
                self.extract_answer(span_type=self.args.span_type)

        except Exception as e:
            logging.error(e)


def main(args):
    try:
        logging.info("Extraction start! Setting up...")
        extractor = AnswerExtractor(args=args)
        logging.info("Setup done! Extracting answers...")
        extractor.extract()
        logging.info("Extraction done!")
    except Exception as e:
        logging.error(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=DATA_DIR, type=str)
    parser.add_argument("--input_file", default=INPUT_FILE, type=str)
    parser.add_argument("--output_dir", default=DATA_DIR, type=str)
    parser.add_argument("--output_file", default=None, type=str)
    parser.add_argument("--stopwords_dir", default=DATA_DIR, type=str)
    parser.add_argument("--stopwords_file", default=STOPWORDS_FILE, type=str)
    parser.add_argument("--span_type", default="ALL", type=str)
    parser.add_argument("--lang", default="vi", type=str)
    parser.add_argument("--use_gpu", default=True, type=bool)
    parser.add_argument("--device", default=0, type=int)
    args = parser.parse_args()

    assert os.path.exists(os.path.join(args.input_dir, args.input_file))
    assert os.path.exists(args.output_dir)
    assert os.path.exists(os.path.join(args.stopwords_dir, args.stopwords_file))

    main(args)
