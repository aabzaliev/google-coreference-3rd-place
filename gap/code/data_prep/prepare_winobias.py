import collections
import glob
import os

import numpy as np
import pandas as pd


def prepare_winonbias(raw_data_dir, processed_data_dir):
    go_through = glob.glob(os.path.join(raw_data_dir, 'winobias_data', '*.txt.*'))

    winobias = None

    for fname in go_through:
        lines = open(fname).readlines()
        if 'anti_stereotyped_type1.txt.dev' in fname:
            # type1 where A and B can be properly deducted
            df = make_gap_like_objects_type1(lines)
        else:
            df = make_gap_like_objects_type2(lines)

        if winobias is None:
            winobias = df
        else:
            winobias = winobias.append(df)

    winobias.to_csv(os.path.join(processed_data_dir, 'winobias_prepared.tsv'), sep='\t', index=False)


def from_word_offset_to_char_offset(w, offsets):
    local_text = w

    word_offset_A = offsets[0][0]
    word_offset_B = offsets[1][0]
    word_offset_P = offsets[2][0]

    # add the len to account for the empty spaces
    char_offset_A = sum([len(i) for i in local_text[:word_offset_A]]) + len(local_text[:word_offset_A])
    char_offset_B = sum([len(i) for i in local_text[:word_offset_B]]) + len(local_text[:word_offset_B])
    char_offset_P = sum([len(i) for i in local_text[:word_offset_P]]) + len(local_text[:word_offset_P])

    return (char_offset_A, char_offset_B, char_offset_P)


def get_word_offsets(tokens):
    clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
    words = []

    for index, token in enumerate(tokens):
        if "[" in token and "]" in token:
            clusters[0].append((index, index))
        elif "[" in token:
            clusters[0].append((index, index))
        elif "]" in token:
            old_span = clusters[0][-1]
            clusters[0][-1] = (old_span[0], index)

        if token.endswith("."):
            # Winobias is tokenised, but not for full stops. (and zapyataya)
            # We'll just special case them here.
            token = token[:-1]
            words.append(token.strip("[]()"))
            words.append(".")
        else:
            words.append(token.strip("[]()"))

    assert len(clusters) == 1
    return words, clusters[0]


def make_gap_like_objects_type1(lines):
    good_rows = list()

    for sentence1, sentence2 in zip(lines[0::2], lines[1::2]):

        obj1 = dict()
        obj2 = dict()

        # sentence1
        tokens1 = sentence1.strip().replace(',', ' ,').split(" ")  # zapyataya is not wokring that good
        _id1 = tokens1.pop(0)

        # get the word offsets and the cleaned text tokens
        w1, offsets1 = get_word_offsets(tokens1)
        A1 = " ".join(w1[offsets1[0][0]:offsets1[0][1] + 1])  #
        P1 = " ".join(w1[offsets1[1][0]:offsets1[1][1] + 1])  # P is different but we can took out from them A and B

        # sentence2
        tokens2 = sentence2.strip().replace(',', ' ,').split(" ")
        _id2 = tokens2.pop(0)
        # get the word offsets and the cleaned text tokens
        w2, offsets2 = get_word_offsets(tokens2)
        B2 = " ".join(w2[offsets2[0][0]:offsets2[0][1] + 1])  #
        P2 = " ".join(w2[offsets2[1][0]:offsets2[1][1] + 1])  # P is different but we can took out from them A and B

        # try to get the position if the word token is the same
        A2 = " ".join(
            w2[offsets1[0][0]:offsets1[0][1] + 1])  # get offsets from the first sentence but take from second text

        if A1 != A2:
            raise NotImplementedError("Was smth wrong ")
            # consider A1 to be the truth (because it was highlighted in the text)

        # now after we found out the B assign the B to the first sentence as well
        # get the offsets from the second sentence but apply to the first one
        B1 = " ".join(w1[offsets2[0][0]:offsets2[0][1] + 1])

        # add the info about A and B to the offsets
        proper_offsets1 = offsets1.copy()
        proper_offsets2 = offsets2.copy()

        proper_offsets1.insert(1, offsets2[
            0])  # the order of offsets is A, B, P so for 1 sentence we insert B (that's why 1)
        proper_offsets2.insert(0, offsets1[
            0])  # and for the second sentence we are looking for A, so insert at the 1st position

        # this is for 2 edge cases
        if B1 != B2:
            # B2 is the truth
            positions = list()
            # adjust the positions of the tokens
            for mini_t in B2.split(' '):
                positions.append(w1.index(mini_t))
            new_offset = (np.min(positions), np.max(positions))

            B1 = " ".join(w1[new_offset[0]:new_offset[1] + 1])
            # readjust the offsets
            proper_offsets1[1] = (new_offset[0], new_offset[1])

        # assert offsets1[1] == offsets2[1] # position of the pronoun is the same. Too restrictive.
        # Sometimes it can be shifted by couple of words. I will check that it is indeed pronoun instead
        assert P1 in ['he', 'she', 'his', 'her', 'him']
        assert P2 in ['he', 'she', 'his', 'her', 'him']
        assert A1 == A2 == " ".join(w1[proper_offsets1[0][0]:proper_offsets1[0][1] + 1]) == " ".join(
            w2[proper_offsets2[0][0]:proper_offsets2[0][1] + 1])
        assert B1 == B2 == " ".join(w1[proper_offsets1[1][0]:proper_offsets1[1][1] + 1]) == " ".join(
            w2[proper_offsets2[1][0]:proper_offsets2[1][1] + 1])

        A_offset_char1, B_offset_char1, Pronoun_offset_char1 = from_word_offset_to_char_offset(w1, proper_offsets1)
        A_offset_char2, B_offset_char2, Pronoun_offset_char2 = from_word_offset_to_char_offset(w2, proper_offsets2)

        # first A is always true, and B we deduce from obj2
        obj1 = {'source': 'winobias', 'Text': " ".join(w1), 'Pronoun': P1, 'Pronoun-offset': Pronoun_offset_char1,
                'A': A1, 'A-offset': A_offset_char1, 'A-coref': True, 'B': B1, 'B-offset': B_offset_char1,
                'B-coref': False, 'URL': ''}

        # second B is always true, and A is deducted
        obj2 = {'source': 'winobias', 'Text': " ".join(w2), 'Pronoun': P2, 'Pronoun-offset': Pronoun_offset_char2,
                'A': A2, 'A-offset': A_offset_char2, 'A-coref': False, 'B': B2, 'B-offset': B_offset_char2,
                'B-coref': True, 'URL': ''}

        good_rows.append(obj1)
        good_rows.append(obj2)

    return pd.DataFrame(good_rows)


def make_gap_like_objects_type2(lines):
    good_rows = list()

    for sentence1, sentence2 in zip(lines[0::2], lines[1::2]):

        obj1 = dict()
        obj2 = dict()

        # sentence1
        tokens1 = sentence1.strip().replace(',', ' ,').split(" ")  # zapyataya is not wokring that good
        _id1 = tokens1.pop(0)

        # get the word offsets and the cleaned text tokens
        w1, offsets1 = get_word_offsets(tokens1)
        A1 = " ".join(w1[offsets1[0][0]:offsets1[0][1] + 1])  #
        P1 = " ".join(w1[offsets1[1][0]:offsets1[1][1] + 1])  # P is different but we can took out from them A and B

        # sentence2
        tokens2 = sentence2.strip().replace(',', ' ,').split(" ")
        _id2 = tokens2.pop(0)
        # get the word offsets and the cleaned text tokens
        w2, offsets2 = get_word_offsets(tokens2)
        B2 = " ".join(w2[offsets2[0][0]:offsets2[0][1] + 1])  #
        P2 = " ".join(w2[offsets2[1][0]:offsets2[1][1] + 1])  # P is different but we can took out from them A and B
        try:
            ##################################################
            #                  Find A2
            ##################################################

            positions = list()
            # adjust the positions of the tokens
            lowered_w2 = [i.lower() for i in w2]
            if 'the' in A1.lower():  # hack because finding the index of 'the is tricky'
                for mini_t in A1.lower().split(' ')[1:]:
                    positions.append(lowered_w2.index(mini_t))
                new_offset = (np.min(positions) - 1, np.max(positions))
            else:
                for mini_t in A1.lower().split(' '):
                    positions.append(lowered_w2.index(mini_t))
                new_offset = (np.min(positions), np.max(positions))
            A2 = " ".join(w2[new_offset[0]:new_offset[1] + 1])
            assert A2.lower() == A1.lower()
            offsets2.insert(0, (new_offset[0], new_offset[1]))

            # now after we found out the B assign the B to the first sentence as well
            # get the offsets from the second sentence but apply to the first one
            #     B1 = " ".join(w1[offsets2[0][0]:offsets2[0][1]+1])

            # add the info about A and B to the offsets
            proper_offsets1 = offsets1.copy()
            proper_offsets2 = offsets2.copy()

            ##########################################
            # Find B1
            ##########################################
            # B2 is the truth
            positions = list()
            # adjust the positions of the tokens
            lowered_w1 = [i.lower() for i in w1]
            if 'the' in B2.lower():
                for mini_t in B2.lower().split(' ')[1:]:
                    positions.append(lowered_w1.index(mini_t))
                new_offset = (np.min(positions) - 1, np.max(positions))
            else:
                for mini_t in B2.lower().split(' '):
                    positions.append(lowered_w1.index(mini_t))
                new_offset = (np.min(positions), np.max(positions))

            B1 = " ".join(w1[new_offset[0]:new_offset[1] + 1])
            proper_offsets1.insert(1, (new_offset[0], new_offset[
                1]))  # the order of offsets is A, B, P so for 1 sentence we insert B (that's why 1)

            # assert offsets1[1] == offsets2[1] # position of the pronoun is the same. Too restrictive.
            # Sometimes it can be shifted by couple of words. I will check that it is indeed pronoun instead
            assert P1 in ['he', 'she', 'his', 'her', 'him']
            assert P2 in ['he', 'she', 'his', 'her', 'him']
            assert A1.lower() == A2.lower() == " ".join(
                w1[proper_offsets1[0][0]:proper_offsets1[0][1] + 1]).lower() == " ".join(
                w2[proper_offsets2[0][0]:proper_offsets2[0][1] + 1]).lower()
            assert B1.lower() == B2.lower() == " ".join(
                w1[proper_offsets1[1][0]:proper_offsets1[1][1] + 1]).lower() == " ".join(
                w2[proper_offsets2[1][0]:proper_offsets2[1][1] + 1]).lower()

            A_offset_char1, B_offset_char1, Pronoun_offset_char1 = from_word_offset_to_char_offset(w1, proper_offsets1)
            A_offset_char2, B_offset_char2, Pronoun_offset_char2 = from_word_offset_to_char_offset(w2, proper_offsets2)

            # first A is always true, and B we deduce from obj2
            obj1 = {'source': 'winobias', 'Text': " ".join(w1), 'Pronoun': P1, 'Pronoun-offset': Pronoun_offset_char1,
                    'A': A1, 'A-offset': A_offset_char1, 'A-coref': True, 'B': B1, 'B-offset': B_offset_char1,
                    'B-coref': False, 'URL': ''}

            # second B is always true, and A is deducted
            obj2 = {'source': 'winobias', 'Text': " ".join(w2), 'Pronoun': P2, 'Pronoun-offset': Pronoun_offset_char2,
                    'A': A2, 'A-offset': A_offset_char2, 'A-coref': False, 'B': B2, 'B-offset': B_offset_char2,
                    'B-coref': True, 'URL': ''}

            good_rows.append(obj1)
            good_rows.append(obj2)
        except:
            print(f"Skipped {_id1} because of some errors!")

    return pd.DataFrame(good_rows)
