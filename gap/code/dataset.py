import numpy as np
import torch
from torch.utils.data import Dataset


def insert_tag(row):
    """Insert custom tags to help us find the position of A, B, and the pronoun after tokenization."""
    to_be_inserted = sorted([
        (row["A-offset"], " [A] "),
        (row["B-offset"], " [B] "),
        (row["Pronoun-offset"], " [P] ")
    ], key=lambda x: x[0], reverse=True)
    text = row["Text"]
    for offset, tag in to_be_inserted:
        text = text[:offset] + tag + text[offset:]
    return text


def tokenize(text, tokenizer):
    """Returns a list of tokens and the positions of A, B, and the pronoun."""
    entries = {}
    final_tokens = []
    for token in tokenizer.tokenize(text):
        if token in ("[A]", "[B]", "[P]"):
            entries[token] = len(final_tokens)
            continue
        final_tokens.append(token)
    return final_tokens, (entries["[A]"], entries["[B]"], entries["[P]"])


class GAPDataset(Dataset):
    """Custom GAP Dataset class"""

    def __init__(self, df, tokenizer, labeled=True):
        self.labeled = labeled
        if labeled:
            tmp = df[["A-coref", "B-coref"]].astype('bool').copy()
            tmp["Neither"] = ~(df["A-coref"] | df["B-coref"])
            self.y = tmp.values.astype("bool")

        # Extracts the tokens and offsets(positions of A, B, and P)
        self.offsets, self.tokens, self.in_urls, self.other_feats = [], [], [], []
        for ix, row in df.iterrows():

            len_pronoun = len(tokenizer.tokenize(row['Pronoun']))
            len_A = len(tokenizer.tokenize(row['A']))
            len_B = len(tokenizer.tokenize(row['B']))

            text = insert_tag(row)
            tokens, offsets = tokenize(text, tokenizer)

            # add the end of the entity to the offsets
            A_dist = abs(offsets[2] - offsets[0])
            B_dist = abs(offsets[2] - offsets[1])
            AB_dist = abs(offsets[1] - offsets[0])
            offsets = [[o, o + l - 1] for o, l in zip(offsets, (len_A, len_B, len_pronoun))]

            # -1 because inclusive bounds! cost me one week of experiments I couldn't understand
            self.offsets.append(offsets)
            if len(tokens) <= 512:
                self.tokens.append(tokenizer.convert_tokens_to_ids(
                    ["[CLS]"] + tokens + ["[SEP]"]))
            else:
                self.tokens.append(tokenizer.convert_tokens_to_ids(
                    ["[CLS]"] + tokens[0:510] + ["[SEP]"]))
                print('Shortened seq')

            self.in_urls.append((row['A_in_URL'], row['B_in_URL']))
            self.other_feats.append((row['A_cont'], row['B_cont'], row['P_cont'], row['A_num'], row['B_num'],
                                     row['P_num'], row['A_rank'], row['B_rank'], row['P_rank'], row['gender'], A_dist,
                                     B_dist, AB_dist, row['synt_dist_A'], row['synt_dist_B']))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.labeled:
            return self.tokens[idx], self.offsets[idx], self.in_urls[idx], self.other_feats[idx], self.y[idx]
        return self.tokens[idx], self.offsets[idx], self.in_urls[idx], self.other_feats[idx], None


def collate_examples(batch, truncate_len=512):  # 512 as in paper
    """Batch preparation.

    1. Pad the sequences
    2. Transform the target.
    """
    transposed = list(zip(*batch))
    max_len = min(
        max((len(x) for x in transposed[0])),
        truncate_len
    )
    tokens = np.zeros((len(batch), max_len), dtype=np.int64)
    for i, row in enumerate(transposed[0]):
        row = np.array(row[:truncate_len])
        tokens[i, :len(row)] = row
    token_tensor = torch.from_numpy(tokens)
    # Offsets
    offsets = torch.stack([
        torch.LongTensor(x) for x in transposed[1]
    ], dim=0) + 1  # Account for the [CLS] token
    # in_urls
    in_urls = torch.stack([
        torch.FloatTensor(x) for x in transposed[2]
    ], dim=0)
    # other features
    other_feats = torch.stack([
        torch.FloatTensor(x) for x in transposed[3]
    ], dim=0)

    # Labels
    if len(transposed) == 4:
        return token_tensor, offsets, in_urls, other_feats, None
    one_hot_labels = torch.stack([
        torch.from_numpy(x.astype("uint8")) for x in transposed[4]
    ], dim=0)
    _, labels = one_hot_labels.max(dim=1)
    return token_tensor, offsets, in_urls, other_feats, labels
