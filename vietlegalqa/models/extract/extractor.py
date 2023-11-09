from typing import Any, List

from stanza.models.common.doc import Document as SDoc
from stanza.models.constituency.parse_tree import Tree as STree
from stanza.pipeline.core import Pipeline as SPipe

from vietlegalqa.data.doc import Document
from vietlegalqa.data.qa import QADataset

from .utils import is_stop


class Extractor:
    def __init__(
        self, document: Document, stopwords: List[str], parser: SPipe, pos: SPipe
    ) -> None:
        self.doc = document
        self.qa = QADataset()
        self.stopwords = stopwords
        self.parser = parser
        self.pos = pos

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def get_answer_start(self, answer: str, question: str, context: list[str]):
        q_tokens = []
        q_nlp = self.pos(question).sentences[0].words
        for word in question:
            if not is_stop(word):
                q_tokens.append(word.lemma)

        ctx_rank = []
        for ctx in context:
            if ctx.find(answer) != -1:
                score = 0
                for words in question:
                    if not self.is_stop(words) and words.lemma in q_tokens:
                        score += 1
                ctx_rank.append({"score": score, "context": ctx})

        answer_start = -1

        if len(ctx_rank) != 0:
            ctx_rank = sorted(ctx_rank, key=lambda x: x["score"], reverse=True)
            refined_answer = ctx_rank[0]["context"]
            answer_start = " ".join(context).find(refined_answer) + refined_answer.find(
                answer
            )

        return answer_start
