from stanza.pipeline.core import Pipeline
from tqdm import tqdm
from typing import List

from vietlegalqa.data.doc import Document
from vietlegalqa.data.qa import QADataset, QAPair

from .utils import *


class QAConstruct:
    def __init__(self, stopwords: List[str], parser: Pipeline, pos: Pipeline) -> None:
        self.data = QADataset()
        self.stopwords = stopwords
        self.parser = parser
        self.pos = pos

    def __call__(
        self, document: Document, id_prefix: Optional[str] = "qa"
    ) -> QADataset:
        for article in tqdm(document, desc="QA Dataset Generation"):
            for summary in article.summary:
                summary, summary_nlp = get_summary_nlp(
                    summary=summary, parser=self.parser
                )

                try:
                    keys = {
                        pos_tag: get_keys(doc_nlp=summary_nlp, pos_tag=pos_tag)
                        for pos_tag in POS_TAGS
                    }
                    keys["NE"] = summary_nlp.ents
                except Exception as e:
                    raise e
                if sum([len(key) for key in keys]) == 0:
                    continue

                try:
                    clauses = extract_clauses(
                        summary_nlp, s_threshold=3, comma_threshold=5
                    )
                except Exception as e:
                    raise e
                if len(clauses) == 0:
                    continue

                for tag, answers in keys.items():
                    if tag == "NE":
                        for answer in summary_nlp.ents:
                            if len(answer.text) == 0:
                                continue

                            questions = [
                                clause.replace(answer.text, answer.type, 1)
                                for clause in clauses
                                if clause.find(answer.text) != -1
                            ]

                            if len(questions) == 0:
                                questions = [
                                    tree_to_text(sent.constituency).replace(
                                        answer.text,
                                        answer.type,
                                        1,
                                    )
                                    for sent in summary_nlp.sentences
                                    if answer.end_char <= sent.tokens[-1].end_char
                                ]

                            if len(questions) == 0:
                                continue
                            question = questions[0]

                            try:
                                context_id, start = get_answer_start(
                                    answer=answer.text,
                                    question=question,
                                    article=article,
                                    pos=self.pos,
                                    stopwords=self.stopwords,
                                )
                            except Exception as e:
                                raise e
                            if start == -1:
                                continue

                            self.data.append(
                                QAPair(
                                    id=f"{id_prefix}_{len(self.data)}",
                                    article=f"{article.id}__{context_id}",
                                    question=question,
                                    answer=answer.text,
                                    start=start,
                                    type=answer.type,
                                    is_impossible=False,
                                )
                            )
                    else:
                        for answer in answers:
                            if len(answer) == 0:
                                continue

                            questions = [
                                clause.replace(answer, POS_REPLACE[tag], 1)
                                for clause in clauses
                                if len(answer) != 0 and clause.find(answer) != -1
                            ]

                            if len(questions) == 0:
                                continue
                            question = questions[0]

                            try:
                                context_id, start = get_answer_start(
                                    answer=answer,
                                    question=question,
                                    article=article,
                                    pos=self.pos,
                                    stopwords=self.stopwords,
                                )
                            except Exception as e:
                                raise e
                            if start == -1:
                                continue

                            self.data.append(
                                QAPair(
                                    id=f"tvpl_{len(self.data)}",
                                    article=f"{article.id}__{context_id}",
                                    question=question,
                                    answer=answer,
                                    start=start,
                                    type=POS_REPLACE[tag],
                                    is_impossible=False,
                                )
                            )

        return self.data
