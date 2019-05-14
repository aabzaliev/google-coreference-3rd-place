import json
import re
import subprocess
from collections import defaultdict
from itertools import chain

import networkx as nx
import nltk
import numpy as np
import pandas as pd
import spacy
from attrdict import AttrDict
from nltk.parse.corenlp import CoreNLPParser
from tqdm import tqdm

nlp = spacy.load('en_core_web_lg')


# NOTE: code is still messy, majority of it is taken from the kaggle kernels
# and from the reproduce_gap_results git repo
# many functions are not even used. TODO: rework

class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}

    def add_edge(self, from_node, to_node, weight, back_penalty=1):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight * back_penalty


def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            raise Exception("Something is wrong")
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []
    dist = 0
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        dist += shortest_paths[current_node][1]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path, dist


def get_rank(token):
    """Step up with token.head until it reaches the root. Returns with step number and root"""
    i = 0
    next_token = token
    while (next_token != next_token.head):
        i += 1
        next_token = next_token.head
    return i, next_token


def child_count(token):
    cc = 0
    for child in token.children:
        cc += 1
    return cc


def build_answers(data):
    answers = []
    for i in range(len(data)):
        dataNext = data.loc[i]
        Acoref = dataNext["A-coref"]
        Bcoref = dataNext["B-coref"]
        answerNext = [int(Acoref), int(Bcoref), 1 - int(Acoref or Bcoref)]
        answers.append(answerNext)
    return np.vstack(answers)


def build_features(data):
    """Generates features from input data"""
    features = []
    sum_good = 0
    for i in range(0, len(data)):
        fi = []
        dataNext = data.loc[i]
        text = dataNext["Text"]
        # print(visualise(dataNext))
        doc = nlp(text)
        Aoff = dataNext["A-offset"]
        Boff = dataNext["B-offset"]
        Poff = dataNext["Pronoun-offset"]
        lth = len(text)

        for token in doc:
            if (token.idx == Aoff):
                Atoken = token
            if (token.idx == Boff):
                Btoken = token
            if (token.idx == Poff):
                Ptoken = token
        Arank, Aroot = get_rank(Atoken)
        Brank, Broot = get_rank(Btoken)
        Prank, Proot = get_rank(Ptoken)

        graph = Graph()

        for token in doc:
            graph.add_edge(token, token.head, 1, 4)

        sent_root = []
        for sent in doc.sents:
            sent_root.append(sent.root)
        for j in range(len(sent_root) - 1):
            graph.add_edge(sent_root[j], sent_root[j + 1], 1, 4)
        try:
            _, Alen = dijsktra(graph, Atoken, Ptoken)
        except:
            Alen = 300
        try:
            _, Blen = dijsktra(graph, Btoken, Ptoken)
        except:
            Blen = 300

        sent_num = len(sent_root)
        for i in range(len(sent_root)):
            if Aroot == sent_root[i]:
                Atop = i
            if Broot == sent_root[i]:
                Btop = i
            if Proot == sent_root[i]:
                Ptop = i

        fi.append(Aoff / lth)  # 0
        fi.append(Boff / lth)  # 1
        fi.append(Poff / lth)  # 2

        fi.append(1.0 * Atop / sent_num)  # 3
        fi.append(1.0 * Btop / sent_num)  # 4
        fi.append(1.0 * Ptop / sent_num)  # 5

        fi.append(Arank / 10)  # 6
        fi.append(Brank / 10)  # 7
        fi.append(Prank / 10)  # 8

        # fi.append(Atoken.similarity(Ptoken))#9
        # fi.append(Btoken.similarity(Ptoken))#10

        # fi.append(Alen/300)#9
        # fi.append(Blen/300)#10

        # fi.append(child_count(Aroot))#11
        # fi.append(child_count(Broot))#12
        # fi.append(child_count(Proot))#13

        features.append(fi)
    return np.vstack(features)


def swap_raws(data, i, j):
    """Swap the ith and jth column of the data"""
    new_data = np.copy(data)
    temp = np.copy(new_data[:, i])
    new_data[:, i] = new_data[:, j]
    new_data[:, j] = temp
    return new_data


# This code is referenced from
# https://www.kaggle.com/keyit92/coref-by-mlp-cnn-coattention

def bs(lens, target):
    low, high = 0, len(lens) - 1

    while low < high:
        mid = low + int((high - low) / 2)

        if target > lens[mid]:
            low = mid + 1
        elif target < lens[mid]:
            high = mid
        else:
            return mid + 1

    return low


def bin_distance(dist):
    buckets = [1, 2, 3, 4, 5, 8, 16, 32, 64]
    low, high = 0, len(buckets)
    while low < high:
        mid = low + int((high - low) / 2)
        if dist > buckets[mid]:
            low = mid + 1
        elif dist < buckets[mid]:
            high = mid
        else:
            return mid

    return low


def distance_features(P, A, B, char_offsetP, char_offsetA, char_offsetB, text, URL):
    doc = nlp(text)

    lens = [token.idx for token in doc]
    mention_offsetP = bs(lens, char_offsetP) - 1
    mention_offsetA = bs(lens, char_offsetA) - 1
    mention_offsetB = bs(lens, char_offsetB) - 1

    mention_distA = mention_offsetP - mention_offsetA
    mention_distB = mention_offsetP - mention_offsetB

    splited_A = str(A).split()[0].replace("*", "")
    splited_B = str(B).split()[0].replace("*", "")

    if re.search(splited_A[0], str(URL)):
        contains = 0
    elif re.search(splited_B[0], str(URL)):
        contains = 1
    else:
        contains = 2

    dist_binA = bin_distance(mention_distA)
    dist_binB = bin_distance(mention_distB)
    output = [dist_binA, dist_binB, contains]

    return output


def extract_dist_features(df):
    index = df.index
    columns = ["D_PA", "D_PB", "IN_URL"]
    dist_df = pd.DataFrame(index=index, columns=columns)

    for i in tqdm(range(len(df))):
        text = df.loc[i, 'Text']
        P_offset = df.loc[i, 'Pronoun-offset']
        A_offset = df.loc[i, 'A-offset']
        B_offset = df.loc[i, 'B-offset']
        P, A, B = df.loc[i, 'Pronoun'], df.loc[i, 'A'], df.loc[i, 'B']
        URL = df.loc[i, 'URL']

        dist_df.iloc[i] = distance_features(P, A, B, P_offset, A_offset, B_offset, text, URL)

    return dist_df


def standford_tokenize(text, a, b, pronoun_offset, a_offset, b_offset, **kwargs):
    # needs to make options work for certain scenarios, might affect downstream applications
    # Example - fractions are not pslit by white-space
    #           but syntactic parsing splits the fractions
    res = model.api_call(text, properties={'annotators': 'tokenize,ssplit'})
    # , 'options': 'tokenizeNLs=true,strictTreebank3=false,normalizeSpace=True'})
    res = AttrDict(res)

    sent_lens = [0] + [len(sent.tokens) for sent in res.sentences]
    sent_lens = np.cumsum(sent_lens)

    # Stanford token indexing start at 1
    # rename keys: index -> i,
    #              characterOffsetBegin -> idx,
    #              originalText -> text
    # rename keys, to induce a uniform api between stanford and spacy
    # remember allennlp and huggingface both use spacy under the hood
    doc = []
    for i, sent in enumerate(res.sentences):
        assert i == sent.index
        for j, token in enumerate(sent.tokens):
            assert j + 1 == token.index

            doc.append(AttrDict({
                'i': token.index + sent_lens[i] - 1,
                'idx': token.characterOffsetBegin,
                'text': token.originalText,
                # word is normalized, list '(' -> '-LRB-'
                # prase tree contains words
                'word': token.word
            }))

    a_end_idx = a_offset + len(a) - 1
    b_end_idx = b_offset + len(b) - 1

    pronoun_offset = map_chars_to_tokens(doc, pronoun_offset)
    pronoun_token = doc[pronoun_offset]

    a_offset = map_chars_to_tokens(doc, a_offset)
    token_end = map_chars_to_tokens(doc, a_end_idx)
    a_span = [a_offset, token_end]
    a_tokens = doc[a_offset:token_end + 1]

    b_offset = map_chars_to_tokens(doc, b_offset)
    token_end = map_chars_to_tokens(doc, b_end_idx)
    b_span = [b_offset, token_end]
    b_tokens = doc[b_offset:token_end + 1]

    tokens = [tok.text for tok in doc]

    return doc, tokens, pronoun_offset, a_offset, b_offset, a_span, b_span, pronoun_token, a_tokens, b_tokens


def map_chars_to_tokens(doc, char_offset):
    #### Convert character level offsets to token level
    # tokenization or mention labelling may not be perfect
    # Identify the token that contains first character of mention
    # Identify the token that contains last character of mention
    # example - token: 'Delia-', mention: 'Delia'
    #           token: 'N.J.Parvathy', mention: 'Parvathy'
    # Token starts before the last character of mention and ends after the last character of mention
    # Remember character offset end here is the character immediately after the token
    return next(filter(lambda token: char_offset in range(token.idx, token.idx + len(token.text)), doc), None).i


def parse_tree_to_graph(sent_trees, doc, **kwargs):
    graph = nx.Graph()
    leaves = []
    edges = []
    for sent_tree in sent_trees:
        edges, leaves = get_edges_in_tree(sent_tree, leaves=leaves, path='', edges=edges, **kwargs)
    graph.add_edges_from(edges)

    tokens = [token.word for token in doc]

    #     assert tokens == leaves, 'Tokens in parse tree and input sentence don\'t match.'

    return graph


# DFS
# trace path to create unique names for all nodes
def get_edges_in_tree(parent, leaves=[], path='', edges=[], lrb_rrb_fix=False):
    for i, node in enumerate(parent):
        if type(node) is nltk.Tree:
            from_node = path
            to_node = '{}-{}-{}'.format(path, node.label(), i)
            edges.append((from_node, to_node))

            if lrb_rrb_fix:
                if node.label() == '-LRB-':
                    leaves.append('(')
                if node.label() == '-RRB-':
                    leaves.append(')')

            edges, leaves = get_edges_in_tree(node, leaves, to_node, edges)
        else:
            from_node = path
            to_node = '{}-{}'.format(node, len(leaves))
            edges.append((from_node, to_node))
            leaves.append(node)
    return edges, leaves


def get_syntactical_distance_from_graph(graph, token_a, token_b, debug=False):
    return nx.shortest_path_length(graph,
                                   source='{}-{}'.format(token_a.word, token_a.i),
                                   target='{}-{}'.format(token_b.word, token_b.i))


def get_normalized_tag(token):
    tag = token.dep_
    tag = 'subj' if 'subj' in tag else tag
    tag = 'dobj' if 'dobj' in tag else tag
    return tag


def two_synt_distances(row):
    doc, tokens, pronoun_offset, a_offset, b_offset, a_span, b_span, pronoun_token, a_tokens, b_tokens = standford_tokenize(
        row['Text'],
        str(row['A']),
        str(row['B']),
        row['Pronoun-offset'],
        row['A-offset'],
        row['B-offset'])

    trees = STANFORD_MODEL.parse_text(row['Text'])
    graph = parse_tree_to_graph(trees, doc)
    try:
        a_synt_dist = get_syntactical_distance_from_graph(graph, a_tokens[0], pronoun_token)
        b_synt_dist = get_syntactical_distance_from_graph(graph, b_tokens[0], pronoun_token)
    except:
        print("Was a mistake returned average!")
        return 11, 11

    return a_synt_dist, b_synt_dist


model = STANFORD_MODEL = CoreNLPParser(url='http://localhost:9090/')


class CoreNLPServer():
    def __init__(self, classpath=None, corenlp_options=None, java_options=['-Xmx5g']):
        self.classpath = classpath
        self.corenlp_options = corenlp_options
        self.java_options = java_options

    def start(self):
        corenlp_options = [('-' + k, str(v)) for k, v in self.corenlp_options.items()]
        corenlp_options = list(chain(*corenlp_options))
        cmd = ['java'] \
              + self.java_options \
              + ['-cp'] \
              + [self.classpath + '*'] \
              + ['edu.stanford.nlp.pipeline.StanfordCoreNLPServer'] \
              + corenlp_options
        self.popen = subprocess.Popen(cmd)
        self.url = 'http://localhost:{}/'.format(self.corenlp_options.port)

    def stop(self):
        self.popen.terminate()
        self.popen.wait()

    # timeout is hardcoded to 60s in nltk implementation
    def api_call(self, data, properties=None, timeout=280):
        default_properties = {
            'outputFormat': 'json',
            'annotators': 'tokenize,pos,lemma,ssplit,{parser_annotator}'.format(
                parser_annotator=self.parser_annotator
            ),
        }

        default_properties.update(properties or {})

        response = self.session.post(
            self.url,
            params={'properties': json.dumps(default_properties)},
            data=data.encode(self.encoding),
            timeout=timeout,
        )

        response.raise_for_status()

        return response.json()


# Instantiate stanford corenlp server
STANFORD_CORENLP_PATH = 'stanford-corenlp-full-2018-10-05/'
server = CoreNLPServer(classpath=STANFORD_CORENLP_PATH,
                       corenlp_options=AttrDict({'port': 9090,
                                                 'timeout': '600000',
                                                 'quiet': 'true',
                                                 'preload': 'tokenize,spplit,lemma,parse,deparse'}))
server.start()
STANFORD_SERVER_URL = server.url
