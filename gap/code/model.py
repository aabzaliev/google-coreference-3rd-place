# more or less the general model used for both pretraining and gap-training

import torch
import torch.nn as nn
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from pytorch_pretrained_bert.modeling import BertModel


class Head(nn.Module):
    """The MLP submodule"""

    def __init__(self, bert_hidden_size: int, cnn_context: int, hidden_size: int):
        super().__init__()
        self.bert_hidden_size = bert_hidden_size
        self.cnn_context = cnn_context
        self.proj_dim = 64
        self.k = 1 + 2 * self.cnn_context
        self.hidden_size = hidden_size

        self.span_extractor = SelfAttentiveSpanExtractor(self.proj_dim)  # span extractor comes directly after BERT
        # all the main parameters are coming from the conv layer
        self.context_conv = nn.Conv1d(self.bert_hidden_size, self.proj_dim, kernel_size=self.k, stride=1,
                                      padding=self.cnn_context, dilation=1, groups=1, bias=True)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.proj_dim * 3),
            #             nn.Dropout(0.7),
            nn.Linear(self.proj_dim * 3, self.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Dropout(0.6),
        )

        self.new_last_layer = nn.Linear(self.hidden_size + 9 + 1 + 3 + 2 + 2, 3)
        # 64 are from proj_dim, 2 are from url, 9 is for the other features, 1 is gender,
        # 3 are synt distance, 2 are the distances to the root

        # after fine-tuning BERT this is not required, throw away
        for i, module in enumerate(self.fc):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                print("Initing batchnorm")
            elif isinstance(module, nn.Linear):
                if getattr(module, "weight_v", None) is not None:
                    nn.init.uniform_(module.weight_g, 0, 1)
                    nn.init.kaiming_normal_(module.weight_v)
                    print("Initing linear with weight normalization")
                    assert model[i].weight_g is not None
                else:
                    nn.init.kaiming_normal_(module.weight)
                    print("Initing linear")
                nn.init.constant_(module.bias, 0)

    def forward(self, bert_outputs, offsets, in_urls, other_feats):
        assert bert_outputs.size(2) == self.bert_hidden_size
        # reduce the dimension
        conv_output = self.context_conv(bert_outputs.transpose(1, 2)).transpose(2, 1).contiguous()
        # and extract the span
        extracted_outputs = self.span_extractor(conv_output, offsets).view(bert_outputs.size(0), -1)
        fc_output = self.fc(extracted_outputs)
        concatenated_outputs = torch.cat([fc_output, in_urls, other_feats], dim=1)
        return self.new_last_layer(concatenated_outputs)


class GAPModel(nn.Module):
    """The main model."""

    def __init__(self, bert_model: str, cnn_context: int, layer: int, hidden_size: int, device: torch.device):
        super().__init__()
        self.device = device
        if bert_model in ("bert-base-uncased", "bert-base-cased"):
            self.bert_hidden_size = 768
        elif bert_model in ("bert-large-uncased", "bert-large-cased"):
            self.bert_hidden_size = 1024
        else:
            raise ValueError("Unsupported BERT model.")
        self.bert = BertModel.from_pretrained(bert_model).to(device)
        self.num_layers = self.bert.config.num_hidden_layers
        self.head = Head(self.bert_hidden_size, cnn_context, hidden_size).to(device)
        self.layer = layer

    def forward(self, token_tensor, offsets, in_urls, other_feats):
        token_tensor = token_tensor.to(self.device)
        bert_outputs, _ = self.bert(
            token_tensor, attention_mask=(token_tensor > 0).long(),
            token_type_ids=None, output_all_encoded_layers=True)
        # calling output_all_encoded_layers False and True with last index is different
        # most likely because of the pooling layer. Without pooling layers slighly better results
        #         h_enc = bert_outputs[-1]
        #         h_lex = self.bert.embeddings.word_embeddings(token_tensor) # this option takes the first part of bert only
        #         h_lex = self.bert.embeddings.LayerNorm(h_lex)
        bert_outputs = bert_outputs[self.layer]

        head_outputs = self.head(bert_outputs, offsets.to(self.device), in_urls.to(self.device),
                                 other_feats.to(self.device))
        return head_outputs


def children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())


def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)


def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))
