import os, json
import argparse
import stanza
from tqdm import tqdm
from underthesea import word_tokenize

DATA_DIR = "./data"
INPUT_FILE = "tvpl"
SPAN_TYPES = ["NE", "NP", "AP", "VP", "S"]
STOPWORDS_FILE = "vietnamese-stopwords.txt"


with open(os.path.join(DATA_DIR, STOPWORDS_FILE), "r", encoding="utf-8") as file:
    STOPWORDS = file.read().splitlines()

PARSER = stanza.Pipeline(
    lang="vi",
    processors="tokenize, pos, constituency",
    use_gpu=True,
    device=0,
    verbose=False,
)
NER = stanza.Pipeline(
    lang="vi",
    processors="tokenize, ner",
    use_gpu=True,
    device=0,
    verbose=False,
)
POS = stanza.Pipeline(
    lang="vi",
    processors="tokenize, pos, lemma",
    use_gpu=True,
    device=0,
    verbose=False,
)


def is_stop(word):
    """Checks if a given word is a stop word. Stop words list is retrieved from:

    Parameters
    ----------
    word
        The parameter "word" is expected to be an object that represents a word.

    Returns
    -------
        A boolean value. If the word is a stop word list, it will return True. Otherwise, it will
    return False.

    """
    if word.text in STOPWORDS:
        return True
    else:
        return False


def extract_s_clause(tree):
    """The function searches for all instances of the "S" label in a given constituency tree and returns a list of those instances.

    Parameters
    ----------
    tree_node
        The parameter `tree_node` is expected to be an instance of `stanza.models.constituency.parse_tree.Tree`. This is a tree structure that represents the parse tree of a sentence or phrase.

    Returns
    -------
        A list of nodes that have the label "S" in the given parse tree.

    """
    if not isinstance(tree, stanza.models.constituency.parse_tree.Tree):
        return []
    clause = []

    for leaf in tree.children:
        clause += extract_s_clause(leaf)

    if tree.label == "S":
        clause.append(tree)

    return clause


def get_answer_clause(sentence, parser):
    parsing_tree = parser(sentence)
    sbar = extract_s_clause(parsing_tree.sentences[0].constituency)[:-1]
    result = []

    for node in sbar:
        if node.label == "S":
            item = " ".join(node.leaf_labels())
            if len(item.split()) <= 5:
                continue
            result.append(item)

    result = sorted(result, key=lambda x: len(x))
    result2 = []
    sentence = " ".join(parsing_tree.sentences[0].constituency.leaf_labels())
    clauses = sentence.split(",")

    for i in range(len(clauses)):
        item, p = clauses[i], i + 1
        while len(item.split()) < 10 and p < len(clauses):
            item = ",".join([item, clauses[p]])
            p += 1
        result2.append(item.strip())

    result2 = sorted(result2, key=lambda x: len(x))
    return result + result2


def get_answer_start(answer, question, sentences, pos):
    q_tokens = []
    q_doc = pos(question).sentences[0].words
    for token in q_doc:
        if not is_stop(token):
            q_tokens.append(token.lemma)

    result = []
    for sent in sentences:
        if sent.find(answer) == -1:
            continue
        sent_doc = pos(sent).sentences[0].words
        score = 0
        for token in sent_doc:
            if is_stop(token):
                continue
            if token.lemma in q_tokens:
                score += 1
        result.append([score, sent])
    if len(result) == 0:
        return -1
    else:
        result = sorted(result, key=lambda x: x[0])
        res_sent = result[0][1]
        answer_start = " ".join(sentences).find(res_sent) + res_sent.find(answer)
        return answer_start


def extract_constituency(tree_node, span_type):
    """search span via span_type from a sentence(parsed as a tree_node)

    Args:
        tree_node (nltk.Tree): a parsed sentence
        span_type (str): the type of the span to be extracted

    Returns:
        list: a list of the satisfying spans having the span_type
    """

    if len(tree_node.children) == 0:
        return []
    spans = []
    children = list(tree_node.children)
    for child in children:
        spans += extract_constituency(child, span_type)
    if span_type == tree_node.label.upper():
        spans.append(" ".join(tree_node.leaf_labels()))
    return spans


def extract_answer_span_constituency(sentence, span_type):
    try:
        constituency_tree = PARSER(sentence).sentences[0].constituency
    except Exception as e:
        return None
    spans = extract_constituency(constituency_tree, span_type)
    return spans


def extract_answer_span(input_data, span_type):
    """Takes in input data and a span type, and extracts answer spans from the input data based on the specified span type (Noun phrase, verb phrase, ADJ/ADV phrase, clause. Returns a cloze-style dataset with questions and answers based on the extracted information.

    Parameters
    ----------
    input_data
        A list of dictionaries. Each dictionary represents a document and contains the following keys: `url`, `title`, `summary`, `document`
    span_type
        Used to specify the type of answer span you want to extract. It could be `NP`, `VP`, `AP`, `S` or any other type of span that was defined in the `ArgumentParser`.

    Returns
    -------
        The function `extract_answer_span` returns the variable `cloze_data`.

    """
    cloze_data = []

    q_count = 0
    c_count = 0

    for item in tqdm(input_data, desc="Answer Extraction"):
        entry = {}
        entry["title"] = item["title"]

        paragraph = {}
        paragraph["summary"] = item["summary"]
        paragraph["context"] = " ".join(item["document"])

        qas = []

        for sent_id, sent in enumerate(item["summary"]):
            spans = extract_answer_span_constituency(sent, span_type)
            if spans is None:
                continue
            try:
                clause = get_answer_clause(sent, PARSER)
            except Exception as e:
                continue

            for ent in spans:
                answer = ent.strip()
                question = None
                for each in clause:
                    if len(answer.split()) >= len(each.split()):
                        continue
                    if each.find(answer) != -1:
                        question = each.replace(answer, "PLACEHOLDER", 1)
                        break
                if not question:
                    continue

                answer_start = get_answer_start(answer, question, item["document"], POS)
                if answer_start == -1:
                    continue

                qas.append(
                    {
                        "id": f'{item["url"]}_{q_count}',
                        "is_impossible": False,
                        "question": question,
                        "answers": [
                            {
                                "answer_start": answer_start,
                                "text": answer,
                                "type": span_type,
                                "sent_id": sent_id,
                            }
                        ],
                    }
                )
                q_count += 1

        paragraph["QA"] = qas
        entry["entry"] = [paragraph]

        cloze_data.append(entry)
        c_count += 1

    print(f"Questions Number: {q_count}")
    return cloze_data


def extract_answer_NE(input_data):
    """Takes in TVPL and extracts answers from it using named entity recognition (NER) and clause parsing techniques. Returns a cloze-style dataset with questions and answers based on the extracted information.

    Parameters
    ----------
    input_data
        A list of dictionaries. Each dictionary represents a document and contains the following keys: `url`, `title`, `summary`, `document`

    Returns
    -------
        Cloze-style dataset generated from the TVPL dataset.

    """
    cloze_data = []

    q_count = 0
    c_count = 0

    for item in tqdm(input_data, desc="Answers Extraction"):
        entry = {}
        entry["title"] = item["title"]

        paragraph = {}
        paragraph["summary"] = item["summary"]
        paragraph["context"] = " ".join(item["document"])

        qas = []

        for sent_id, sent in enumerate(item["summary"]):
            sent = " ".join(word_tokenize(sent))
            sent_doc = NER(sent).ents

            try:
                clause = get_answer_clause(sent, PARSER)
            except Exception as e:
                continue

            for ent in sent_doc:
                answer = ent.text
                question = None

                for each in clause:
                    if each.find(answer) != -1:
                        question = each.replace(answer, ent.type, 1)
                        break
                    else:
                        question = sent[: ent.start_char] + sent[
                            ent.start_char :
                        ].replace(answer, ent.type, 1)

                if not question:
                    continue

                answer_start = get_answer_start(answer, question, item["document"], POS)
                if answer_start == -1:
                    continue

                qas.append(
                    {
                        "id": f'{item["url"]}_{q_count}',
                        "is_impossible": False,
                        "question": question,
                        "answers": [
                            {
                                "answer_start": answer_start,
                                "text": answer,
                                "type": ent.type,
                                "sent_id": sent_id,
                            }
                        ],
                    }
                )
                q_count += 1

        paragraph["QA"] = qas
        entry["entry"] = [paragraph]

        cloze_data.append(entry)
        c_count += 1

    print(f"Questions Number: {q_count}")
    return cloze_data


def extract_answer(args, input_data: list, span_type: str):
    """Based on the value of `span_type`, calls either `extract_answer_NE` or `extract_answer_span` to extract the answer from the `input_data`. The extracted answer is then dumped into a JSON file.

    Parameters
    ----------
    input_data : `list`
        A list of dictionaries. Each dictionary represents a document and contains the following keys: `url`, `title`, `summary`, `document`
    span_type : `str`
        Used to specify the type of answer span you want to extract. It could be `NP`, `VP`, `AP`, `S` or any other type of span that was defined in the `ArgumentParser`.

    """
    print(f"Span type: {span_type}")
    if span_type == "NE":
        cloze_data = extract_answer_NE(input_data)
    else:
        cloze_data = extract_answer_span(input_data, span_type)

    json.dump(
        cloze_data,
        open(
            os.path.join(
                args.output_dir,
                f"{args.input_dir}_answer_extract_{span_type}.json",
            ),
            "w",
            encoding="utf-8",
        ),
        indent=4,
        ensure_ascii=False,
    )


def main(args):
    """The main function, reads an input file, and then calls the extract_answer function for each specified span type."""

    with open(
        os.path.join(args.input_dir, args.input_file), "r", encoding="utf-8"
    ) as file:
        input_data = json.load(file)

    for span_type in [args.span_type]:
        extract_answer(args, input_data, span_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=DATA_DIR, type=str)
    parser.add_argument("--input_file", default=INPUT_FILE, type=str)
    parser.add_argument("--output_dir", default=DATA_DIR, type=str)
    parser.add_argument("--span_type", choices=SPAN_TYPES, type=str)
    args = parser.parse_args()
    assert os.path.exists(os.path.join(args.input_dir, args.input_file))
    assert os.path.exists(args.output_dir)
    main(args)
