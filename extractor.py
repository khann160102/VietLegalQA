import argparse, json, logging, os
import stanza

from stanza.models.common.doc import Document
from stanza.models.constituency.parse_tree import Tree
from torch.cuda import is_available, device_count, get_device_name
from tqdm import tqdm

import utils

if not os.path.exists(os.path.join(utils.LOG_DIR)):
    os.makedirs(os.path.join(utils.LOG_DIR))

logging.basicConfig(
    format=utils.LOG_FORMAT,
    datefmt=utils.DATE_FORMAT,
    level=logging.INFO,
    encoding="utf-8",
    handlers=[
        logging.FileHandler(os.path.join(utils.LOG_DIR, utils.LOG_FILE)),
        logging.StreamHandler(),
    ],
)


class AnswerExtractor:
    def __init__(self, args) -> None:
        """Constructor of the `AnswerExtractor` class. Initializes main components, loads data and NLP models.

        Parameters
        ----------
        args
            Contains arguments from ArgumentParser in the `main()` function.
        """
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
                allow_unknown_language=True,
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
                allow_unknown_language=True,
            )
        except Exception as e:
            logging.error(e)
        logging.info("POS Tagger loaded!")

    def is_stop(self, word: str) -> bool:
        """Checks if a given word is a stop word.

        Parameters
        ----------
        word : str
            The string that represents a word.

        Returns
        -------
            A boolean value. If the word is a stop word, it will return True. Otherwise, it will return False.
        """
        return True if word in self.STOPWORDS else False

    def get_answer_start(self, answer: str, question: str, context: list[str]) -> int:
        """Takes an answer, question, and context as input and returns the starting index of the answer within the context.

        Parameters
        ----------
        answer : str
            A string that represents the answer to a question.
        question : str
            A string that represents the question being asked.
        context : list[str]
            A list of strings that represents the context or passage from which the question is being asked. It provides the necessary information or background for answering the question.

        Returns
        -------
            A integer value. Returns the starting index of the answer within the context.
        """
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

    def extract_s_clauses(self, node: Tree, threshold: int == 3) -> list[str]:
        """Recursively extracts clauses from a parsing tree if the node label is "S" and the number of leafs is greater than a given threshold.

        Parameters
        ----------
        node : stanza.models.constituency.parse_tree.Tree
            Represents a constituency tree node.
        threshold : int == 3
            The threshold parameter is an optional integer parameter that determines the minimum number of leaf labels required for a clause to be extracted. The default value for the threshold is 3.

        Returns
        -------
            A list of clauses, presenting as strings.
        """
        if node.is_leaf():
            return []
        clauses = []

        for leaf in node.children:
            clauses += self.extract_s_clauses(leaf, threshold=threshold)

        if node.label == "S" and len(node.leaf_labels()) > threshold:
            clauses.append(" ".join(node.leaf_labels()).strip())

        return clauses

    def extract_comma_clauses(self, tree: Tree) -> list[str]:
        """Returns a list of comma-separated clauses extracted from the given constituency tree.

        Parameters
        ----------
        tree : stanza.models.constituency.parse_tree.Tree
            Represents a constituency tree node.

        Returns
        -------
            A list of comma-separated clauses extracted from the constituency tree, presenting as strings.
        """
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

    def extract_clauses(self, nlp: Document) -> list[str]:
        """Takes a NLP document as input and returns a list of clauses extracted from the document.

        Parameters
        ----------
        nlp : stanza.models.common.doc.Document
            Representing a document with has been processed with Stanza parser.

        Returns
        -------
            A list of strings, which contains the extracted clauses from the input document.
        """
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

    def extract_pos(self, node: Tree, pos_tag: str) -> list[str]:
        """Recursively extracts phrases from a parse tree that have a specific part-of-speech tag.

        Parameters
        ----------
        node : stanza.models.constituency.parse_tree.Tree
            Represents a constituency tree node.
        pos_tag : str
            Represents a part-of-speech tag, must be within the POS tags of Stanza library, which are [`NP`, `VP`, `AP`, `S`]

        Returns
        -------
            A list of strings, which contains the extracted phrases from the input document based on the given POS tag.
        """
        if node.is_leaf():
            return []

        spans = []
        for child in node.children:
            spans += self.extract_pos(child, pos_tag)

        if pos_tag == node.label.upper():
            spans.append(" ".join(node.leaf_labels()))

        return spans

    def extract_pos_answers(self, nlp: Document, pos_tag: str) -> list[str]:
        """Takes a NLP document and a part-of-speech tag as input, and returns a list of spans that match the given part-of-speech tag.

        Parameters
        ----------
        nlp : stanza.models.common.doc.Document
            Representing a document with has been processed with Stanza parser.
        pos_tag : str
            Represents a part-of-speech tag which needed to be extracted, must be within the POS tags of Stanza library, which are [`NP`, `VP`, `AP`, `S`]

        Returns
        -------
            A list of strings, which contains the extracted POS phrases from the input document.
        """
        spans = []
        try:
            for sent in nlp.sentences:
                spans += self.extract_pos(sent.constituency, pos_tag)
        except Exception as e:
            return []

        return spans

    def get_summary(self, doc: str) -> tuple[str, Document]:
        """Takes a string document as input, processes it using the Stanza NLP parser, and returns a tuple containing the document and the processed document itself.

        Parameters
        ----------
        doc : str
            The `doc` parameter is a string that represents the document or text for which you want to processed.

        Returns
        -------
        tuple[str, stanza.models.common.doc.Document]
            A tuple containing two elements. The first element is a string called "doc" which is the input document itself. The second element is the result of processing the input document using the Stanza NLP parser.
        """
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
        answer_id: str,
    ) -> list[dict]:
        """Return a formatted list of dictionaries from the given input.

        Parameters
        ----------
        id : str
            The unique identifier for the question-answer pair.
        is_impossible : bool
            A boolean value indicating whether the question has an answer or not.
        question : str
            The question being asked.
        answer : str
            The answer to the question.
        answer_type : str
            Type of the answer span. It can be either `NE` or within the POS tag of Stanza library [`NP`, `VP`, `AP`, `S`].
        answer_start : int
            The index position where the answer starts in the context or passage.
        answer_id: str
            The `answer_id` parameter is used to uniquely identify the answer within a question-answering dataset.
        """
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
        """Extracts answers from a given dataset by iterating through the `data, extracting summaries and questions, and finding the answer positions based on the specified span type.

        Parameters
        ----------
        span_type : str
            A string that specifies the type of answer span to extract. Represents a part-of-speech tag, must be within the POS tags of Stanza library, which are [`NP`, `VP`, `AP`, `S`]

        Returns
        -------
            A list of dictionaries, where each dictionary represents a cloze-style question and answer.
        """
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

    def extract_answer_ne(self) -> list:
        """Extracts answers from a given dataset by analyzing summaries and identifying relevant clauses and named entities.

        Returns
        -------
            A list of dictionaries, where each dictionary represents a cloze-style question and answer.
        """
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

    def to_json(self, data: list[tuple], span_type: str) -> None:
        """Saves the data as a JSON file.

        Parameters
        ----------
        data : list[tuple]
            A list of tuples. Each tuple represents a piece of data that will be converted to JSON format.
        span_type : str
            Represents the type of span being extracted. It is used to generate the filename for the JSON file that will be saved.

        """
        try:
            filename = os.path.join(
                self.args.output_dir,
                f"{self.args.input_file}_extract_{span_type}.json",
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

    def extract_answer(self, span_type: str) -> None:
        """Extracts answers based on the given span type.

        Parameters
        ----------
        span_type : str
            The `span_type` parameter is a string that specifies the type of span to extract. It can be either `NE` or within the POS tag of Stanza library [`NP`, `VP`, `AP`, `S`].

        """
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

    def extract(self) -> None:
        """The function extracts answer based on the specified span type or all span types if "ALL" is specified."""
        try:
            if self.args.span_type.upper() == "ALL":
                logging.info("Extracting all span types...")
                for type in utils.SPAN_TYPES:
                    self.extract_answer(span_type=type)
            elif self.span_type in utils.SPAN_TYPES:
                logging.info(f"Extracting {self.span_type} span types...")
                self.extract_answer(span_type=self.args.span_type)

        except Exception as e:
            logging.error(e)


def check_args(args) -> tuple[bool, str]:
    if not os.path.exists(os.path.join(args.input_dir)):
        return (False, "Input directory does not exist!")
    if not args.input_file.endswith(".json"):
        setattr(args, "input_file", f"{args.input_file}.json")
    if not os.path.exists(os.path.join(args.input_dir, args.input_file)):
        return (False, "Input file does not exist!")

    if not os.path.exists(os.path.join(args.output_dir)):
        return (False, "Output directory does not exist!")
    if args.output_file.endswith(".json"):
        setattr(args, "output_file", args.output_file.replace(".json", ""))

    if not os.path.exists(os.path.join(args.stopwords_dir)):
        return (False, "Stop words directory does not exist!")
    if not os.path.exists(os.path.join(args.stopwords_dir, args.stopwords_file)):
        return (False, "Stop words file does not exist!")

    if isinstance(args.span_type, str):
        if args.span_type != "ALL" and not args.span_type in utils.SPAN_TYPES:
            return (False, f"{args.span_type} span type is not supported!")

    msg = []
    if args.use_gpu is True:
        if not is_available:
            setattr(args, "use_gpu", False)
            msg.append("CUDA is not available!!! Using CPU instead!!!")
        else:
            msg.append("CUDA available!")

    if args.device >= device_count() or args.device < 0:
        setattr(args, "device", 0)
        msg.append("Device out of index! Reverted to default!")
    msg.append(f"Using device: {get_device_name(args.device)}")

    return True, msg


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
    parser.add_argument("--input_dir", default=utils.INPUT_DIR, type=str)
    parser.add_argument("--input_file", default=utils.INPUT_FILE, type=str)
    parser.add_argument("--output_dir", default=utils.OUTPUT_DIR, type=str)
    parser.add_argument("--output_file", default=utils.OUTPUT_FILE, type=str)
    parser.add_argument("--stopwords_dir", default=utils.STOPWORDS_DIR, type=str)
    parser.add_argument("--stopwords_file", default=utils.STOPWORDS_FILE, type=str)
    parser.add_argument("--span_type", default="ALL", type=str)
    parser.add_argument("--lang", default="vi", type=str)
    parser.add_argument("--use_gpu", default=True, type=bool)
    parser.add_argument("--device", default=0, type=int)
    args = parser.parse_args()

    if check_args(args)[0]:
        for msg in check_args(args)[1]:
            logging.info(msg)
        main(args)
    else:
        logging.error(check_args(args)[1])
