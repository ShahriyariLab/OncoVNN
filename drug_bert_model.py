# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 16:28:23 2022

@author: toxicant
"""
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
import numpy as np
import torch
from tqdm import tqdm

class LigandTokenizer:
    def __init__(self,**config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['seq_model'])

    def tokenize(self, smiles, only_tokens=False):
        # truncation=True enforces the 512-token limit inside the tokenizer,
        # which correctly handles BPE tokens (not raw characters).
        tokens = self.tokenizer(smiles, truncation=True, max_length=512)
        if not only_tokens:
            return tokens
        return list(np.array(tokens['input_ids'], dtype="int"))

def embed_ligand(smiles, chem_tokenizer, lig_embedder):
    tokens = chem_tokenizer.tokenize(smiles, False)
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    # Truncate at token level (not character level), preserving [CLS] and [SEP]
    max_tokens = 512
    if len(input_ids) > max_tokens:
        input_ids = input_ids[:max_tokens - 1] + [input_ids[-1]]
        attention_mask = [1] * max_tokens

    model_device = next(lig_embedder.parameters()).device
    input_ids_t = torch.LongTensor([input_ids]).to(model_device)
    attn_mask_t = torch.LongTensor([attention_mask]).to(model_device)

    try:
        output = lig_embedder(input_ids_t, attention_mask=attn_mask_t, return_dict=True)
    except Exception:
        print(smiles)
        raise

    last_hidden = output.last_hidden_state[0]   # [seq_len, hidden]
    mask = torch.tensor(attention_mask, dtype=torch.bool)

    # Exclude [CLS] (index 0) and [SEP] (last attended token) from mean pooling
    mask[0] = False
    real_indices = mask.nonzero(as_tuple=True)[0]
    if len(real_indices) > 0:
        mask[real_indices[-1]] = False          # zero out [SEP]

    if mask.sum() == 0:
        mask = torch.tensor(attention_mask, dtype=torch.bool)   # fallback: CLS+SEP only

    return last_hidden[mask].mean(dim=0).cpu().detach().numpy()

# embedded_drug = {}
# chem_tokenizer = LigandTokenizer()
# lig_embedder = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
# lig_embedder.eval()
# print('Getting embedding SMILES feature')

# for smiles in tqdm(train_set['smiles']):
#     embedded_drug[smiles] = embed_ligand(smiles, chem_tokenizer, lig_embedder)