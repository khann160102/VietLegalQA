from typing import Dict, List

from stanza.pipeline.core import Pipeline as SPipe

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


def is_stop(self, word: str, stopwords: List[str]) -> bool:
    return True if word in stopwords else False


def get_answer_start(answer: str, question: str, context: list[str], pos: SPipe):
    q_nlp = pos(question).sentences[0].tokens
    q_tokens = []
    for word in q_nlp:
        if not is_stop(word):
            q_tokens.append(word.lemma)

    ctx_rank = []
    for ctx in context:
        ctx_nlp = pos(ctx).sentences
        if ctx.find(answer) != -1:
            score = 0
            for words in q_nlp:
                if not is_stop(words) and words.lemma in q_tokens:
                    score += 1
            ctx_rank.append({"score": score, "id": ctx})

    answer_start = -1

    if len(ctx_rank) != 0:
        ctx_rank = sorted(ctx_rank, key=lambda x: x["score"], reverse=True)
        refined_answer = ctx_rank[0]["context"]
        answer_start = " ".join(context).find(refined_answer) + refined_answer.find(
            answer
        )

    return answer_start
