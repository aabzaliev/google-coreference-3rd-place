import json
import os

import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from dataset import GAPDataset, collate_examples
from model import set_trainable, GAPModel
from pretrain import pretrain, load_data_for_pretrain
from train_helpers import GAPBot, TriangularLR


def load_gap_train_data(settings):
    # this data is the same for all 4 models
    train = pd.read_pickle(os.path.join(settings['PROCESSED_DATA_DIR'], 'train.pkl'))
    val = pd.read_pickle(os.path.join(settings['PROCESSED_DATA_DIR'], 'val.pkl'))
    test = pd.read_pickle(os.path.join(settings['PROCESSED_DATA_DIR'], 'test.pkl'))

    gap_train = train.append(val, ignore_index=True, verify_integrity=True, sort=False). \
        append(test, ignore_index=True, verify_integrity=True, sort=False)

    return gap_train


def train(gap_train, cased, layer, h_layer_size, seed, cv_seed, settings, lr=2e-3):
    '''

    :param gap_train:
    :param cased:
    :param layer:
    :param h_layer_size:
    :param seed:
    :param cv_seed: is required to reconstruct the split during the prediction - UPD: actually not
    :param settings:
    :return:
    '''
    # TODO: do I need to set up this seed every time?
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # shuffle eafter setting the proper seed
    gap_train = gap_train.sample(frac=1.0)

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

    # take the one from the fine-tuned
    best_from_warmup = './models/pretrain/' + model_name + '_finetuned/' + 'best.pth'

    n_folds = 10
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)
    val_scores = list()

    # this is stratify during the cross-validation
    strat = np.zeros(shape=(len(gap_train)))
    strat[gap_train['A-coref']] = 1
    strat[gap_train['B-coref']] = 2

    # TODO: make CV a separate function
    for fold_n, (train_index, valid_index) in enumerate(folds.split(gap_train, y=strat)):
        print("fold n#{}".format(fold_n))
        train_ds = GAPDataset(gap_train.iloc[train_index], tokenizer)
        val_ds = GAPDataset(gap_train.iloc[valid_index], tokenizer)

        train_loader = DataLoader(
            train_ds,
            collate_fn=collate_examples,
            batch_size=20,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
            drop_last=False
        )

        val_loader = DataLoader(
            val_ds,
            collate_fn=collate_examples,
            batch_size=64,
            num_workers=2,
            pin_memory=True,
            shuffle=True
        )

        model = GAPModel(BERT_MODEL, 0, layer, h_layer_size, torch.device("cuda:0"))
        set_trainable(model.bert, False)
        set_trainable(model.head, True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # TODO: no need to create each time separate folder, but different checkpoints
        bot = GAPBot(
            model, train_loader, val_loader,
            optimizer=optimizer, echo=True,
            avg_window=25, checkpoint_dir="./models/gap/" + model_name + '/' + str(fold_n) + '/'
        )

        # Load the best checkpoint from warmup
        bot.load_model(best_from_warmup)

        steps_per_epoch = len(train_loader) * 2
        n_steps = steps_per_epoch * 5

        bot.train(
            3100,
            log_interval=20,
            snapshot_interval=100,  # check the performance every epoch
            scheduler=TriangularLR(
                optimizer, 20, ratio=2, steps_per_cycle=n_steps)
        )

        # Load the best checkpoint
        bot.load_model(bot.best_performers[0][1])

        # eval on the validation dataset
        val_sc = bot.eval(val_loader)
        val_scores.append(val_sc)
        print(val_sc)


def lighten_the_models(settings):
    '''
    we store a lot of unnecessary information during the GAP training, the finetuned BERT embeddings are all the same
    so we resaved alll of them by storing only head
    TODO: do the right saving during the training process
    '''
    # first get all the models
    models = os.listdir(settings['MODELS_GAP_DIR'])
    for model_name in models:
        # and for each fold
        for fold_n in range(10):
            path_to_load = os.path.join(settings['MODELS_GAP_DIR'], model_name, str(fold_n), 'best.pth')
            path_to_resave = os.path.join(settings['MODELS_GAP_DIR'], model_name, str(fold_n), 'light.pth')
            state_dict = torch.load(path_to_load)
            lighter_state_dict = {k: v for k, v in state_dict.items() if k.startswith('head')}
            torch.save(lighter_state_dict, path_to_resave)


if __name__ == '__main__':
    with open('settings.json') as f:
        settings = json.loads(f.read())

    # load the data
    warmup_train, warmup_val = load_data_for_pretrain(settings)
    gap_train = load_gap_train_data(settings)

    #############################################################
    #                      uncased 5
    ##############################################################

    layer = -5
    cased = False
    seed = 1499
    h_layer_size = 64
    bert_finetuning_lr = 2e-5
    pretrain(warmup_train, warmup_val, cased, layer, h_layer_size, seed, bert_finetuning_lr)
    train(gap_train, cased, layer, h_layer_size, seed, 2711, settings)

    #########################################################
    #                     uncased 6
    #########################################################

    layer = -6
    cased = False
    seed = 789
    h_layer_size = 100
    bert_finetuning_lr = 9e-5
    bert_finetuning_snaphot_inerval = 200
    pretrain(warmup_train, warmup_val, cased, layer, h_layer_size, seed, bert_finetuning_lr,
             bert_finetuning_snaphot_inerval)
    train(gap_train, cased, layer, h_layer_size, seed, 98765, settings)

    #############################################################
    #                       cased 5
    #############################################################

    layer = -5
    cased = True
    seed = 1499
    h_layer_size = 64
    bert_finetuning_lr = 4e-5
    pretrain(warmup_train, warmup_val, cased, layer, h_layer_size, seed, bert_finetuning_lr)
    train(gap_train, cased, layer, h_layer_size, seed, 2711, settings)

    #############################################################
    #                         cased6
    #############################################################

    layer = -6
    cased = True
    seed = 16
    h_layer_size = 64
    bert_finetuning_lr = 4e-5
    pretrain(warmup_train, warmup_val, cased, layer, h_layer_size, seed, bert_finetuning_lr)
    train(gap_train, cased, layer, h_layer_size, seed, 60, settings)
