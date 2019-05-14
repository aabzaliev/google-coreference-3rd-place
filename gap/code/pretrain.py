import os

import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import DataLoader

from dataset import GAPDataset, collate_examples
from model import GAPModel, set_trainable
from train_helpers import GAPBot, TriangularLR


# TODO: add settings
def pretrain(warmup_train, warmup_val, cased, layer, h_layer_size, seed, bert_finetuning_lr,
             bert_finetuning_snaphot_inerval=400):
    '''
    pre-trains the networks on ontonotes, dpr and winogender
    validate on
    :param cased - boolean, True - cased version of BERT, FALSE - uncased
    :param layer - integer, -5 or -6
    :return:
    '''

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # shuffle
    warmup_train = warmup_train.sample(frac=1.0)
    warmup_val = warmup_val.sample(frac=1.0)

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

    train_warmup_ds = GAPDataset(warmup_train, tokenizer)
    val_warmup_ds = GAPDataset(warmup_val, tokenizer)

    train_warmup_loader = DataLoader(
        train_warmup_ds,
        collate_fn=collate_examples,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        drop_last=False
    )

    val_warmup_loader = DataLoader(
        val_warmup_ds,
        collate_fn=collate_examples,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        shuffle=True
    )

    # TODO: don't reload but just continue the training
    # the current setup was done because I had 2 models for submissions
    model = GAPModel(BERT_MODEL, 0, layer, h_layer_size, torch.device("cuda:0"))
    set_trainable(model.bert, False)
    set_trainable(model.head, True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # this is always 1e-3!

    bot = GAPBot(
        model, train_warmup_loader, val_warmup_loader,
        optimizer=optimizer, echo=True,
        avg_window=25, checkpoint_dir="./models/pretrain/" + model_name
    )

    bot.train(
        20000,
        log_interval=100,
        snapshot_interval=1000,
        scheduler=TriangularLR(
            optimizer, 20, ratio=2, steps_per_cycle=len(train_warmup_loader) * 3)
    )

    # after training the head unfreeze the given layer and continue training
    layer_idx_to_unfreeze = len(model.bert.encoder.layer) + layer  # layer is negative
    assert layer_idx_to_unfreeze == 19 or layer_idx_to_unfreeze == 18

    model = GAPModel(BERT_MODEL, 0, layer, h_layer_size, torch.device("cuda:0"))
    set_trainable(model.bert, False)
    set_trainable(model.head, False)
    set_trainable(model.bert.encoder.layer[layer_idx_to_unfreeze], True)
    optimizer = torch.optim.Adam(model.parameters(), lr=bert_finetuning_lr)

    bot = GAPBot(
        model, train_warmup_loader, val_warmup_loader,
        optimizer=optimizer, echo=True,
        avg_window=25, checkpoint_dir="./models/pretrain/" + model_name + '_finetuned'
    )

    best_from_warmup = "./models/pretrain/" + model_name + '/best.pth'
    bot.load_model(best_from_warmup)

    bot.train(
        20000,
        log_interval=100,
        snapshot_interval=bert_finetuning_snaphot_inerval,
        scheduler=TriangularLR(
            optimizer, 20, ratio=2, steps_per_cycle=len(train_warmup_loader) * 3)
    )


def load_data_for_pretrain(settings):
    # this data is the same for all 4 models
    dpr = pd.read_pickle(os.path.join(settings['PROCESSED_DATA_DIR'], 'dpr.pkl'))
    winobias = pd.read_pickle(os.path.join(settings['PROCESSED_DATA_DIR'], 'winobias.pkl'))
    winogender = pd.read_pickle(os.path.join(settings['PROCESSED_DATA_DIR'], 'winogender.pkl'))
    ontonotes = pd.read_pickle(os.path.join(settings['PROCESSED_DATA_DIR'], 'ontonotes.pkl'))

    warmup_train = dpr.append(ontonotes, ignore_index=True, verify_integrity=True, sort=False). \
        append(winogender, ignore_index=True, verify_integrity=True, sort=False)
    warmup_val = winobias

    return warmup_train, warmup_val
