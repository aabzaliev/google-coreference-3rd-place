import json
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_prep.prepare_dpr import prepare_dpr
from data_prep.prepare_winobias import prepare_winonbias
from data_prep.prepare_winogender import prepare_winogender
from feature_utils import build_features, two_synt_distances

with open('settings.json') as f:
    settings = json.loads(f.read())


def prepare_all_data(settings):
    prepare_dpr(settings['RAW_DATA_DIR'], settings['PROCESSED_DATA_DIR'])
    prepare_winonbias(settings['RAW_DATA_DIR'], settings['PROCESSED_DATA_DIR'])
    prepare_winogender(settings['RAW_DATA_DIR'], settings['PROCESSED_DATA_DIR'])


def make_features(settings):
    '''

    reads from the preprocessed data folder and makes the features
    :param settings:
    :return:
    '''

    PRONOUNS = {
        'she': "FEMININE",
        'her': "FEMININE",
        'hers': "FEMININE",
        'he': "MASCULINE",
        'his': "MASCULINE",
        'him': "MASCULINE",
    }

    # data from kaggle
    train = pd.read_csv(os.path.join(settings['RAW_DATA_DIR'], 'gap-test_corrected.tsv'), sep='\t')
    val = pd.read_csv(os.path.join(settings['RAW_DATA_DIR'], 'gap-validation_corrected.tsv'), sep='\t')
    test = pd.read_csv(os.path.join(settings['RAW_DATA_DIR'], 'gap-development_corrected.tsv'), sep='\t')
    private_test = pd.read_csv(os.path.join(settings['RAW_DATA_DIR'], 'test_stage_2.tsv'), sep='\t')

    # external data
    dpr = pd.read_csv(os.path.join(settings['PROCESSED_DATA_DIR'], 'dpr_prepared.tsv'), sep='\t')
    winobias = pd.read_csv(os.path.join(settings['PROCESSED_DATA_DIR'], 'winobias_prepared.tsv'), sep='\t')
    winogender = pd.read_csv(os.path.join(settings['PROCESSED_DATA_DIR'], 'winogender_prepared.tsv'), sep='\t')
    ontonotes = pd.read_csv(os.path.join(settings['PROCESSED_DATA_DIR'], 'ontonotes_label_strat.tsv'), sep='\t')

    # TODO: put it into the ontonotes preparation section
    # NA as country is interpreted as nan somewhere
    ontonotes['A'][pd.isnull(ontonotes['A'])] = 'NA'
    ontonotes['B'][pd.isnull(ontonotes['B'])] = 'NA'

    all_dfs = [train, val, test, private_test, dpr, winobias, winogender, ontonotes]
    all_df_names = ['train', 'val', 'test', 'private_test', 'dpr', 'winobias', 'winogender', 'ontonotes']
    print("Feature building started ", time.ctime())

    for df, df_name in tqdm(zip(all_dfs, all_df_names), total=len(all_dfs)):

        # features from the public kernels
        spacy_feat = pd.DataFrame(build_features(df))
        assert spacy_feat.shape[1] == 9
        spacy_feat.columns = ['A_cont', 'B_cont', 'P_cont', 'A_num', 'B_num', 'P_num', 'A_rank', 'B_rank', 'P_rank']

        # syntactic distance with the standford corenlp
        feats = list()
        for ix, row in tqdm(df.iterrows()):
            feats.append(two_synt_distances(row))

        synt_dists = pd.DataFrame(np.asarray(feats))
        synt_dists.columns = ['synt_dist_A', 'synt_dist_B']

        # add spacy and synt distances to the dataframe
        df = pd.concat([df, spacy_feat, synt_dists], axis=1)

        # gender feature
        df['gender'] = df['Pronoun'].apply(lambda x: PRONOUNS.get(x.lower(), 'other')) == 'FEMININE'

        # URL feature
        # for kaggle datasets we have the URL,
        if any(pd.notnull(df['URL'])):
            df['URL'] = df['URL'].apply(lambda x: x.split('wiki/')[1])
        # for external no URL info is available
        else:
            df['URL'].fillna('', inplace=True)

        df['A_in_URL'] = df[['A', 'URL']].apply(lambda x: x[0].replace(" ", "_").lower() in x[1].lower(),
                                                axis=1).astype(np.int8)
        df['B_in_URL'] = df[['B', 'URL']].apply(lambda x: x[0].replace(" ", "_").lower() in x[1].lower(),
                                                axis=1).astype(np.int8)
        # save
        df.to_pickle(os.path.join(settings['PROCESSED_DATA_DIR'], df_name + '.pkl'))


if __name__ == '__main__':
    prepare_all_data(settings)
    make_features(settings)
