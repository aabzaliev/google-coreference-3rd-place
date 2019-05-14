import json
import os

import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import DataLoader

from dataset import GAPDataset, collate_examples
from model import GAPModel
from train_helpers import GAPBot


# there is a bug in the data loader, this is a dirty trick to fix it
def load_gap_test_data(settings):
    gap_test = pd.read_pickle(os.path.join(settings['PROCESSED_DATA_DIR'], 'private_test.pkl'))
    gap_test['A-coref'] = -999
    gap_test['B-coref'] = -999

    return gap_test


def predict(gap_test, cased, layer, h_layer_size, seed, cv_seed, settings):
    # TODO: do I need to set up this seed every time?
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # TODO: as this stuff is part of the each model it can be put into separate function
    assert layer == -5 or layer == -6
    assert isinstance(cased, bool)

    if cased:
        BERT_MODEL = 'bert-large-cased'
    else:
        BERT_MODEL = 'bert-large-uncased'

    if BERT_MODEL.endswith('uncased'):
        DO_LOWER_CASE = True
    elif BERT_MODEL.endswith('cased'):
        DO_LOWER_CASE = False
    else:
        raise ValueError("Improper bert model name!")

    model_name = BERT_MODEL + "_" + str(abs(layer))

    # initialize the tokenizer
    # TODO: out into separate folder
    tokenizer = BertTokenizer.from_pretrained(
        BERT_MODEL,
        do_lower_case=DO_LOWER_CASE,
        never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[A]", "[B]", "[P]")
    )

    # These tokens are not actually used, so we can assign arbitrary values.
    tokenizer.vocab["[A]"] = -1
    tokenizer.vocab["[B]"] = -1
    tokenizer.vocab["[P]"] = -1

    test_ds = GAPDataset(gap_test, tokenizer)
    test_loader = DataLoader(
        test_ds,
        collate_fn=collate_examples,
        batch_size=20,
        num_workers=2,
        pin_memory=True,
        shuffle=False
    )

    model = GAPModel(BERT_MODEL, 0, layer, h_layer_size, torch.device("cuda:0"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # TODO: fix this
    bot = GAPBot(
        model, test_loader, test_loader,
        optimizer=optimizer, echo=True,
        avg_window=25, checkpoint_dir="../blablabla"
    )

    # create the directory for predictions
    pred_dir = os.path.join(settings['PREDICTIONS_DIR'], model_name)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    finetuned_embedding_path = os.path.join(settings['MODELS_PRETRAIN_DIR'], model_name + '_finetuned', 'best.pth')
    # first load the embeddings
    bot.load_model(finetuned_embedding_path)

    # iterate over all cv models and make the predictions
    for fold_n in range(10):
        # Load the best checkpoint from warmup
        model_from_fold = os.path.join(settings['MODELS_GAP_DIR'], model_name, str(fold_n), 'light.pth')
        bot.load_model(model_from_fold)

        # out-of-fold predictions
        preds = bot.predict(test_loader)

        # save them
        np.save(os.path.join(pred_dir, 'fold_' + str(fold_n) + '.csv'), preds.numpy())


if __name__ == '__main__':
    with open('settings.json') as f:
        settings = json.loads(f.read())

    gap_test = load_gap_test_data(settings)

    cv_seed = 98765
    layer = -6
    cased = False
    seed = 789
    h_layer_size = 100
    predict(gap_test, cased, layer, h_layer_size, seed, cv_seed, settings)

    #############################################################
    #                      uncased 5
    ##############################################################

    layer = -5
    cased = False
    seed = 1499
    cv_seed = 2711
    h_layer_size = 64
    predict(gap_test, cased, layer, h_layer_size, seed, cv_seed, settings)

    #############################################################
    #                       cased 5
    #############################################################

    layer = -5
    cased = True
    seed = 1499
    h_layer_size = 64
    cv_seed = 2711
    predict(gap_test, cased, layer, h_layer_size, seed, cv_seed, settings)

    #############################################################
    #                         cased6
    #############################################################

    layer = -6
    cased = True
    seed = 16
    h_layer_size = 64
    cv_seed = 60
    predict(gap_test, cased, layer, h_layer_size, seed, cv_seed, settings)
