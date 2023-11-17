from stanza.models.common.doc import Document, Sentence, Word
from stanza.models.constituency.parse_tree import Tree
from stanza.pipeline.core import Pipeline

from typing import Dict, List, Optional, Tuple
from underthesea import sent_tokenize, word_tokenize

from vietlegalqa.data.doc import Article


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


def tree_to_text(tree: Tree, sep: Optional[str] = " "):
    return sep.join(tree.leaf_labels())


def stanza_tokenizer(text: str) -> str:
    return [word_tokenize(sent) for sent in sent_tokenize(text)]


def get_summary_nlp(
    summary: str,
    parser: Pipeline,
    word_sep: Optional[str] = " ",
    sent_sep: Optional[str] = " ",
) -> Tuple[str, Document]:
    summary_nlp: Document = parser(stanza_tokenizer(text=summary))
    summary = sent_sep.join(
        [
            tree_to_text(tree=sent.constituency, sep=word_sep)
            for sent in summary_nlp.sentences
        ]
    )

    return summary, summary_nlp


def get_pos(node: Tree, pos_tag: str, sep: Optional[str] = " ") -> List[str]:
    if node.is_leaf():
        return list()

    keys: List[str] = list()

    if pos_tag == node.label.upper():
        keys.append(tree_to_text(tree=node, sep=sep))

    for child in node.children:
        keys.extend(get_pos(node=child, pos_tag=pos_tag, sep=sep))

    return keys


def get_keys(doc_nlp: Document, pos_tag: str) -> List[str]:
    keys: List[str] = list()

    try:
        sent: Sentence
        for sent in doc_nlp.sentences:
            keys.extend(get_pos(node=sent.constituency, pos_tag=pos_tag))
    except Exception as e:
        raise e

    return keys


def extract_clauses_constituent(
    node: Tree, threshold: Optional[int] == 3, sep: Optional[str] = " "
) -> List[str]:
    if node.is_leaf():
        return list()

    clauses: List[str] = list()

    if node.label == "S" and len(node.leaf_labels()) > threshold:
        clauses.append(tree_to_text(tree=node, sep=sep))

    for child in node.children:
        clauses.extend(
            extract_clauses_constituent(node=child, threshold=threshold, sep=sep)
        )

    return clauses


def extract_clauses_comma(sent: str, threshold: Optional[int] = 5) -> List[str]:
    clauses: List[str] = list()
    comma_clauses = [clause.strip() for clause in sent.split(",")]

    if len(comma_clauses) > 1:
        for idx, clause in enumerate(comma_clauses):
            for next_clause in enumerate(comma_clauses[idx + 1 :]):
                if len(clause.split()) < threshold:
                    clause += f" , {next_clause}"
                else:
                    break

            clauses.append(clause)

    return clauses


def extract_clauses(
    nlp: Document,
    s_threshold: Optional[int] = 3,
    comma_threshold: Optional[int] = 5,
    sep: Optional[str] = " ",
) -> List[str]:
    clauses: List[str] = list()

    sent: Sentence
    for sent in nlp.sentences:
        try:
            clauses.extend(
                extract_clauses_constituent(
                    node=sent.constituency,
                    threshold=s_threshold,
                    sep=sep,
                )
            )
        except Exception as e:
            raise e

        try:
            clauses.extend(
                extract_clauses_comma(
                    sent=sent.text,
                    threshold=comma_threshold,
                )
            )
        except Exception as e:
            raise e

    return sorted(clauses, key=lambda clause: len(clause), reverse=True)


def is_stop(word: str, stopwords: List[str]) -> bool:
    return True if word in stopwords else False


def get_answer_start(
    answer: str, question: str, article: Article, pos: Pipeline, stopwords: List[str]
) -> Tuple[int, int]:
    question: List[Word] = pos(stanza_tokenizer(question)).sentences[0].words

    q_tokens: List[str] = [
        word.lemma for word in question if not is_stop(word=word, stopwords=stopwords)
    ]

    context_rank: List[Dict[str, int]] = [
        {
            "id": idx,
            "score": len([None for word in q_tokens if word in ctx]),
            "start": ctx.find(answer),
        }
        for idx, ctx in enumerate(article.context)
    ]

    context_rank = sorted(context_rank, key=lambda ctx: ctx["score"], reverse=True)

    return (context_rank[0]["id"], context_rank[0]["start"])
