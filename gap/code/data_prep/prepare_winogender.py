import pandas as pd
import os
import os

import pandas as pd


def prepare_winogender(raw_data_dir, processed_data_dir):

    templates_path = os.path.join(raw_data_dir, 'winogender_templates.tsv')
    S = load_templates(templates_path)

    good_rows = list()
    for s in S:
        occupation, other_participant, answer, sentence = s
        # the position of the words are always the same
        male_sent, female_sent, neutral_sent, A_word_idx, \
        B_word_idx, P_word_idx, female_P, male_P, neutral_P = generate(occupation, other_participant, answer, sentence)

        if A_word_idx == 0:
            A = 'The ' + occupation
        else:
            A = 'the ' + occupation

        B = 'the ' + other_participant
        if B_word_idx == 0:
            B = 'The ' + other_participant
        else:
            B = 'the ' + other_participant

        # can be done just for one sentence, but just to make sure
        A_char_idx, B_char_idx, P_char_idx = from_word_offset_to_char_offset(neutral_sent,
                                                                             (A_word_idx, B_word_idx, P_word_idx))
        #     A_char_idx_female, B_char_idx_female, P_char_idx_female = from_word_offset_to_char_offset(female_sent, (A_word_idx, B_word_idx, P_word_idx))
        #     A_char_idx_male, B_char_idx_male, P_char_idx_male = from_word_offset_to_char_offset(male_sent, (A_word_idx, B_word_idx, P_word_idx))
        #     assert A_char_idx_neutral == A_char_idx_female == A_char_idx_male
        #     assert B_char_idx_neutral == B_char_idx_female == B_char_idx_male
        #     assert P_char_idx_neutral == P_char_idx_female == P_char_idx_male

        male_sentid, female_sentid, neutral_sentid = [
            occupation + '.' + other_participant + '.' + str(answer) + '.' + gender + ".txt" for gender in
            ["male", "female", "neutral"]]

        if answer:
            A_coref = False
            B_coref = True
        elif answer == 0:
            A_coref = True
            B_coref = False

        # male object
        obj_male = {'source': 'winogender', 'Text': male_sent, 'Pronoun': male_P, 'Pronoun-offset': P_char_idx,
                    'A': A, 'A-offset': A_char_idx, 'A-coref': A_coref, 'B': B, 'B-offset': B_char_idx,
                    'B-coref': B_coref, 'URL': ''}

        # female object
        obj_female = {'source': 'winogender', 'Text': female_sent, 'Pronoun': female_P, 'Pronoun-offset': P_char_idx,
                      'A': A, 'A-offset': A_char_idx, 'A-coref': A_coref, 'B': B, 'B-offset': B_char_idx,
                      'B-coref': B_coref, 'URL': ''}

        # neutral obj
        obj_neutral = {'source': 'winogender', 'Text': neutral_sent, 'Pronoun': neutral_P, 'Pronoun-offset': P_char_idx,
                       'A': A, 'A-offset': A_char_idx, 'A-coref': A_coref, 'B': B, 'B-offset': B_char_idx,
                       'B-coref': B_coref, 'URL': ''}

        good_rows.append(obj_male)
        good_rows.append(obj_female)
        good_rows.append(obj_neutral)

        winogender = pd.DataFrame(good_rows)
        winogender.to_csv(os.path.join(processed_data_dir, 'winogender_prepared.tsv'), sep='\t', index=False)


def load_templates(path):
    fp = open(path, 'r')
    S = []
    headers = next(fp).strip().split('\t')
    for line in fp:
        line = line.strip().split('\t')
        occupation, other_participant, answer, sentence = line[0], line[1], int(line[2]), line[3]
        S.append((occupation, other_participant, answer, sentence))
    return S


def from_word_offset_to_char_offset(w, offsets):
    local_text = w.split(" ")

    word_offset_A = offsets[0]
    word_offset_B = offsets[1]
    word_offset_P = offsets[2]

    # add the len to account for the empty spaces
    char_offset_A = sum([len(i) for i in local_text[:word_offset_A]]) + len(local_text[:word_offset_A])
    char_offset_B = sum([len(i) for i in local_text[:word_offset_B]]) + len(local_text[:word_offset_B])
    char_offset_P = sum([len(i) for i in local_text[:word_offset_P]]) + len(local_text[:word_offset_P])

    return (char_offset_A, char_offset_B, char_offset_P)


def generate(occupation, other_participant, answer, sentence, someone=False, context=None):
    # use the random male and female names from wikipedia to generate the sentences
    toks = sentence.split(" ")
    occ_index = toks.index("$OCCUPATION")
    A_word_idx = occ_index - 1  # because of the
    part_index = toks.index("$PARTICIPANT")
    B_word_idx = part_index - 1  # because of the
    toks[occ_index] = occupation
    if not someone:  # we are using the instantiated participant, e.g. "client", "patient", "customer",...
        toks[part_index] = other_participant
    else:  # we are using the bleached NP "someone" for the other participant
        # first, remove the token that precedes $PARTICIPANT, i.e. "the"
        toks = toks[:part_index - 1] + toks[part_index:]
        # recompute participant index (it should be part_index - 1)
        part_index = toks.index("$PARTICIPANT")
        if part_index == 0:
            toks[part_index] = "Someone"
        else:
            toks[part_index] = "someone"
    NOM = "$NOM_PRONOUN"
    POSS = "$POSS_PRONOUN"
    ACC = "$ACC_PRONOUN"
    special_toks = set({NOM, POSS, ACC})
    P_word_idx = [toks.index(i) for i in special_toks if i in toks]
    assert len(P_word_idx) == 1
    P_word_idx = P_word_idx[0]

    female_map = {NOM: "she", POSS: "her", ACC: "her"}
    male_map = {NOM: "he", POSS: "his", ACC: "him"}
    neutral_map = {NOM: "they", POSS: "their", ACC: "them"}

    female_toks = [x if not x in special_toks else female_map[x] for x in toks]
    female_P = [female_map[x] for x in toks if x in special_toks]

    male_toks = [x if not x in special_toks else male_map[x] for x in toks]
    male_P = [male_map[x] for x in toks if x in special_toks]

    neutral_toks = [x if not x in special_toks else neutral_map[x] for x in toks]
    neutral_P = [neutral_map[x] for x in toks if x in special_toks]

    male_sent, female_sent, neutral_sent = " ".join(male_toks), " ".join(female_toks), " ".join(neutral_toks)
    neutral_sent = neutral_sent.replace("they was", "they were")
    neutral_sent = neutral_sent.replace("They was", "They were")
    return male_sent, female_sent, neutral_sent, A_word_idx, B_word_idx, P_word_idx, female_P[0], male_P[0], neutral_P[
        0]