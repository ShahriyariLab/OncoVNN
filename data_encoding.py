# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel
from data_loader import GetData
from drug_bert_model import LigandTokenizer, embed_ligand
from tqdm import tqdm

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_EMBED_CACHE_PATH = "drug_embeddings_cache.pkl"

class DataEncoding:
    def __init__(self, vocab_dir):
        self.vocab_dir = vocab_dir
        self.Getdata = GetData()

        print(f"[DataEncoding] Loading ChemBERTa on {_DEVICE} ...")
        self.tokenizer = LigandTokenizer(seq_model="seyonec/PubChem10M_SMILES_BPE_450k")
        self.model = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k").to(_DEVICE)
        self.model.eval()
        print("[DataEncoding] ChemBERTa ready.")

    def _embed_smiles(self, smile):
        return embed_ligand(smile, self.tokenizer, self.model)

    def _load_cache(self):
        if os.path.exists(_EMBED_CACHE_PATH):
            import pickle
            with open(_EMBED_CACHE_PATH, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_cache(self, cache):
        import pickle
        with open(_EMBED_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)

    def encode(self, traindata, testdata):
        drug_smiles = self.Getdata.getDrug()
        drugid2smile = dict(zip(drug_smiles['drug_id'], drug_smiles['smiles']))
        traindata['smiles'] = [drugid2smile[i] for i in traindata['DRUG_ID']]
        testdata['smiles'] = [drugid2smile[i] for i in testdata['DRUG_ID']]

        unique_smiles = list(drug_smiles['smiles'].unique())
        cache = self._load_cache()
        missing = [s for s in unique_smiles if s not in cache]
        if missing:
            print(f"[DataEncoding] Embedding {len(missing)} new SMILES (cached: {len(cache)}) ...")
            for smi in tqdm(missing, desc="ChemBERTa", unit="mol"):
                cache[smi] = self._embed_smiles(smi)
            self._save_cache(cache)
            print(f"[DataEncoding] Cache saved to {_EMBED_CACHE_PATH}")
        else:
            print(f"[DataEncoding] All {len(unique_smiles)} SMILES loaded from cache.")
        embedded_smiles = cache

        traindata['drug_encoding'] = [embedded_smiles[s] for s in traindata['smiles']]
        testdata['drug_encoding'] = [embedded_smiles[s] for s in testdata['smiles']]

        assert all(len(enc) == 768 for enc in traindata['drug_encoding']), "Train drug encoding size mismatch"
        assert all(len(enc) == 768 for enc in testdata['drug_encoding']), "Test drug encoding size mismatch"

        rna_df = pd.read_csv('GDSC_data/filtered_gene_expression.txt', delimiter='\t')

        rna_df = rna_df.set_index('GENE_SYMBOLS')
        rna_df = rna_df.drop(columns=['GENE_title'])

        rna_df.columns = rna_df.columns.str.replace('DATA.', '', regex=False)

        available_rna_cosmics = set(rna_df.columns)

        train_cosmic_ids = [cid for cid in traindata['COSMIC_ID'].astype(str).unique() if cid in available_rna_cosmics]
        test_cosmic_ids = [cid for cid in testdata['COSMIC_ID'].astype(str).unique() if cid in available_rna_cosmics]

        if len(train_cosmic_ids) < len(traindata['COSMIC_ID'].unique()):
            print(f"[INFO] Filtered out {len(traindata['COSMIC_ID'].unique()) - len(train_cosmic_ids)} train COSMIC_IDs not in RNA data.")

        if len(test_cosmic_ids) < len(testdata['COSMIC_ID'].unique()):
            print(f"[INFO] Filtered out {len(testdata['COSMIC_ID'].unique()) - len(test_cosmic_ids)} test COSMIC_IDs not in RNA data.")

        train_rnadata_full = rna_df[train_cosmic_ids]
        test_rnadata_full = rna_df[test_cosmic_ids]

        train_rnadata = train_rnadata_full.T
        test_rnadata = test_rnadata_full.T

        train_rnadata.columns = train_rnadata.columns.astype(str)
        test_rnadata.columns = test_rnadata.columns.astype(str)

        train_rnadata.index.name = 'COSMIC_ID'
        test_rnadata.index.name = 'COSMIC_ID'

        return traindata, train_rnadata, testdata, test_rnadata


if __name__ == '__main__':
    vocab_dir = 'DeepTTC'
    obj = DataEncoding(vocab_dir=vocab_dir)
    traindata, testdata = obj.Getdata.ByCancer(random_seed=1)

    traindata, train_rnadata, testdata, test_rnadata = obj.encode(
        traindata=traindata,
        testdata=testdata)

    print(traindata.head())
    print(train_rnadata.head())
    print(testdata.head())
    print(test_rnadata.head())
