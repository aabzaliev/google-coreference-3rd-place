import glob
import json
import logging as log
import os
import re

import pandas as pd
from sacremoses import MosesTokenizer

TOKENIZER = MosesTokenizer()

# NOTE: some of the DPR preparation code is taken from
# https://github.com/nyu-mll/jiant

def prepare_dpr(raw_data_dir, processed_data_dir):
    # create the directory for intermediate dpr .jsons
    dpr_jsons_dir = os.path.join(raw_data_dir, 'dpr_jsons')
    if not os.path.exists(dpr_jsons_dir):
        os.makedirs(dpr_jsons_dir)

    src_file = os.path.join(raw_data_dir, "dpr_data.txt")

    # load everything in memory
    # this preprocessing is required to be compatible with the jiant code
    text2examples = {}
    curr = {}
    with open(src_file) as fd:
        for line in fd:
            line = line.strip()
            if not line:
                curr_text = curr["text"]
                if curr_text not in text2examples:
                    text2examples[curr_text] = []
                text2examples[curr_text].append(curr)
                # make new curr
                curr = {}
            else:
                # get id # here we add the information, eahc text is repeated two times - TRUE and FALSE
                line = line.split(":")
                key = line[0].strip()
                val = " ".join(line[1:]).strip()
                curr[key] = val

    # I already have them split
    # save them in the raw folder again
    split_files = {k: open(os.path.join(dpr_jsons_dir, f"{k}.json"), 'w')
                   for k in ["train", "dev", "test"]}

    skip_counter = 0
    for text, example in text2examples.items():
        record = convert_text_examples_to_json(text, example)
        if not record.get("targets", []):
            skip_counter += 1
            continue
        # Write to file by split key.
        split = record["info"]["split"]
        split_files[split].write(json.dumps(record))
        split_files[split].write("\n")

    log.info("Skipped %d examples with no targets found.", skip_counter)
    file_inputs = glob.glob(dpr_jsons_dir + '/*.json')

    good_rows = list()
    for fname in file_inputs:
        inputs = list(load_lines(fname))
        for line in inputs:
            line = json.loads(line)
            if len(line['targets']) > 1:
                good_rows.append(make_row(line))

    dpr_df = pd.DataFrame(good_rows)
    dpr_df.to_csv(os.path.join(processed_data_dir, 'dpr_prepared.tsv'), sep='\t', index=False)


def convert_text_examples_to_json(text, example):
    # This assert makes sure that no text appears in train and test
    tokens = TOKENIZER.tokenize(text)
    split = set([ex['partof'] for ex in example])
    assert len(split) == 1
    obj = {"text": " ".join(tokens),
           "info": {'split': list(split)[0],
                    'source': 'recast-dpr'},
           "targets": []
           }
    for ex in example:
        hyp = TOKENIZER.tokenize(ex['hypothesis'])
        assert len(tokens) <= len(hyp)
        found_diff_word = False
        for idx, pair in enumerate(zip(tokens, hyp)):
            if pair[0] != pair[1]:
                referent = ''
                found_diff_word = True
                distance = len(hyp) - len(tokens) + 1
                pro_noun = tokens[idx]
                found_referent = False
                for word_idx in range(idx + 1):
                    referent = hyp[idx:idx + distance]
                    if word_idx == 0:
                        referent[0] = referent[0][0].upper() + referent[0][1:]
                    if referent == tokens[word_idx:word_idx + distance]:
                        found_referent = True
                        target = {'span1': [idx, idx + 1],
                                  'span2': [word_idx, word_idx + distance],
                                  'label': ex['entailed'],
                                  'span1_text': pro_noun,
                                  'span2_text': " ".join(tokens[word_idx:word_idx + distance])
                                  }
                        obj['targets'].append(target)
                        break;
                break;

    return obj


def from_word_offset_to_char_offset(line):
    local_text = line['text'].split(' ')

    word_offset_A = line['targets'][0]['span2'][0]
    word_offset_B = line['targets'][1]['span2'][0]
    word_offset_P = line['targets'][0]['span1'][0]

    # add the len to account for the empty spaces
    char_offset_A = sum([len(i) for i in local_text[:word_offset_A]]) + len(local_text[:word_offset_A])
    char_offset_B = sum([len(i) for i in local_text[:word_offset_B]]) + len(local_text[:word_offset_B])
    char_offset_P = sum([len(i) for i in local_text[:word_offset_P]]) + len(local_text[:word_offset_P])

    return (char_offset_A, char_offset_B, char_offset_P)


def make_row(line):
    Text = line['text']
    source = line['info']['split']

    # were checked before that for pronoun span1 = span2
    Pronoun = line['targets'][0]['span1_text']
    A = line['targets'][0]['span2_text']
    B = line['targets'][1]['span2_text']

    Pronoun_offset = [m.start() for m in re.finditer(Pronoun, Text)]
    A_offset = [m.start() for m in re.finditer(A, Text)]
    B_offset = [m.start() for m in re.finditer(B, Text)]

    A_offset_char, B_offset_char, Pronoun_offset_char = from_word_offset_to_char_offset(line)

    if len(A_offset) == 1:
        assert A_offset[0] == A_offset_char
        A_offset = A_offset_char
    else:
        A_offset = A_offset_char
    if len(B_offset) == 1:
        assert B_offset[0] == B_offset_char
        B_offset = B_offset_char
    else:
        B_offset = B_offset_char
    if len(Pronoun_offset) == 1:
        assert Pronoun_offset[0] == Pronoun_offset_char
        Pronoun_offset = Pronoun_offset_char
    else:
        Pronoun_offset = Pronoun_offset_char

    len_Pronoun = len(Pronoun)
    len_A = len(A)
    len_B = len(B)

    A_coref = True if line['targets'][0]['label'] == 'entailed' else False
    B_coref = True if line['targets'][1]['label'] == 'entailed' else False
    #     print(Pronoun, Pronoun_offset)
    #     print(A, A_offset, A_coref)
    #     print(B, B_offset, B_coref)

    return {'source': source, 'Text': Text, 'Pronoun': Pronoun, 'Pronoun-offset': Pronoun_offset,
            'A': A, 'A-offset': A_offset, 'A-coref': A_coref, 'B': B, 'B-offset': B_offset, 'B-coref': B_coref,
            'URL': ''}


# def load_json_data(filename: str):
#     ''' Load JSON records, one per line. '''
#     with open(filename, 'r') as fd:
#         for line in fd:
#             yield json.loads(line)


def load_lines(filename: str):
    ''' Load text data, yielding each line. '''
    with open(filename) as fd:
        for line in fd:
            yield line.strip()
