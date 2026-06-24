# python3
# -*- coding:utf-8 -*-

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import csv
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

class GetData():
    def __init__(self):
        PATH = 'GDSC_data'

        rnafile = PATH + '/filtered_gene_expression.txt'
        smilefile = PATH + '/smile_inchi.csv'
        pairfile = PATH + '/GDSC2_fitted_dose_response_25Feb20.xlsx'
        drug_infofile = PATH + "/Drug_listTue_Aug10_2021.csv"
        drug_thred = PATH + '/IC50_thred.txt'

        self.pairfile = pairfile
        self.drugfile = drug_infofile
        self.rnafile = rnafile
        self.smilefile = smilefile
        self.drug_thred = drug_thred

    def getDrug(self):
        """Load and validate drug SMILES data"""
        try:
            drugdata = pd.read_csv(self.smilefile, index_col=0)

            invalid_smiles = drugdata[drugdata['smiles'].isna() |
                                    (drugdata['smiles'].str.strip() == '')]
            if not invalid_smiles.empty:
                print(f"\nWarning: Found {len(invalid_smiles)} drugs with missing/invalid SMILES:")
                print(invalid_smiles[['drug_id', 'Name']].to_string())

            drugdata = drugdata.dropna(subset=['smiles'])
            drugdata = drugdata[drugdata['smiles'].str.strip() != '']

            print(f"\nLoaded {len(drugdata)} valid drug SMILES strings")
            return drugdata

        except Exception as e:
            print(f"Error loading drug SMILES data: {e}")
            return pd.DataFrame()

    def _filter_pair(self, drug_cell_df):
        print("#"*50)
        print("step1 Filter cell lines....")
        not_index = ['908134', '1789883', '908120', '908442']
        print(drug_cell_df.shape)
        drug_cell_df = drug_cell_df[~drug_cell_df['COSMIC_ID'].isin(not_index)]
        print(drug_cell_df.shape)

        print("step2 filter drugs....")
        pub_df = pd.read_csv(self.drugfile)
        pub_df = pub_df.dropna(subset=['PubCHEM'])
        pub_df = pub_df[(pub_df['PubCHEM'] != 'none') & (pub_df['PubCHEM'] != 'several')]
        print(drug_cell_df.shape)
        drug_cell_df = drug_cell_df[drug_cell_df['DRUG_ID'].isin(pub_df['drug_id'])]
        print(drug_cell_df.shape)
        return drug_cell_df

    def _stat_cancer(self, drug_cell_df):
        print("#" * 50)
        cancer_num = drug_cell_df['TCGA_DESC'].value_counts().shape[0]
        print('#\tThere are a total of cancer types: {}'.format(cancer_num))
        min_cancer_drug = min(drug_cell_df['TCGA_DESC'].value_counts())
        max_cancer_drug = max(drug_cell_df['TCGA_DESC'].value_counts())
        mean_cancer_drug = np.mean(drug_cell_df['TCGA_DESC'].value_counts())
        print('#\tThe least cancer type corresponds to {} drugs,\n\tThe maximum number of drugs is {},\n\tThe average number of drugs is {}'.format(
            min_cancer_drug, max_cancer_drug, mean_cancer_drug))

    def _stat_cell(self, drug_cell_df):
        print("#" * 50)
        cell_num = drug_cell_df['COSMIC_ID'].value_counts().shape[0]
        print('#\t The cell lines used are: {}'.format(cell_num))
        min_drug = min(drug_cell_df['COSMIC_ID'].value_counts())
        max_drug = max(drug_cell_df['COSMIC_ID'].value_counts())
        mean_drug = np.mean(drug_cell_df['COSMIC_ID'].value_counts())
        print('#\t The cell line with least number of drugs correspond to {},\n\t The maximum number of drugs is {},\n\t The average number of drugs is {}'.format(
            min_drug, max_drug, mean_drug))

    def _stat_drug(self, drug_cell_df):
        print("#" * 50)
        drug_num = drug_cell_df['DRUG_ID'].value_counts().shape[0]
        print('#\t The drugs used are: {}'.format(drug_num))
        min_cell = min(drug_cell_df['DRUG_ID'].value_counts())
        max_cell = max(drug_cell_df['DRUG_ID'].value_counts())
        mean_cell = np.mean(drug_cell_df['DRUG_ID'].value_counts())
        print('#\t The least number of drugs correspond to {} cell lines,\n\t The maximum number of corresponding cell lines is {},\n\t The average corresponds to {} cell lines'.format(
            min_cell, max_cell, mean_cell))

    def _split(self, df, col, ratio, random_seed):
        col_list = df[col].value_counts().index
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        for instatnce in col_list:
            sub_df = df[df[col] == instatnce]
            sub_df = sub_df[['DRUG_ID', 'COSMIC_ID', 'TCGA_DESC', 'LN_IC50']]
            sub_train, sub_test = train_test_split(sub_df, test_size=ratio, random_state=random_seed)
            if train_data.shape[0] == 0:
                train_data = sub_train
                test_data = sub_test
            else:
                train_data = pd.concat([train_data, sub_train])
                test_data = pd.concat([test_data, sub_test])
        print('#' * 50)
        print('#\t There are a total of data pairs: {}'.format(df.shape[0]))
        print('#\t The training data are: {}'.format(train_data.shape[0]))
        print('#\t The test data are: {}'.format(test_data.shape[0]))

        return train_data, test_data

    def ByCancer(self, random_seed):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)

        drug_cell_df = drug_cell_df.head(10000)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        print(drug_cell_df['TCGA_DESC'].value_counts())

        train_data, test_data = self._split(df=drug_cell_df, col='TCGA_DESC',
                                            ratio=0.2, random_seed=random_seed)

        return train_data, test_data

    def ByDrug(self):
        drug_cell_df = pd.read_excel(self.pairfile)
        drug_cell_df = self._filter_pair(drug_cell_df)
        train_data, test_data = self._split(df=drug_cell_df, col='DRUG_ID', ratio=0.2)
        return train_data, test_data

    def ByCell(self):
        drug_cell_df = pd.read_excel(self.pairfile)
        drug_cell_df = self._filter_pair(drug_cell_df)
        train_data, test_data = self._split(df=drug_cell_df, col='COSMIC_ID', ratio=0.2)
        return train_data, test_data

    def getRna(self, traindata, testdata):
        """Load RNA expression data for training and test sets"""
        rnadata = pd.read_csv(self.rnafile, sep='\t')

        train_cosmic_ids = traindata['COSMIC_ID'].astype(str)
        test_cosmic_ids = testdata['COSMIC_ID'].astype(str)

        train_rnaid = ['DATA.' + str(i) for i in train_cosmic_ids]
        test_rnaid = ['DATA.' + str(i) for i in test_cosmic_ids]

        valid_train_cols = [col for col in train_rnaid if col in rnadata.columns]
        valid_test_cols = [col for col in test_rnaid if col in rnadata.columns]

        train_rnadata = rnadata[valid_train_cols]
        test_rnadata = rnadata[valid_test_cols]

        train_cosmic_map = {f'DATA.{cid}': cid for cid in train_cosmic_ids}
        test_cosmic_map = {f'DATA.{cid}': cid for cid in test_cosmic_ids}

        train_rnadata.columns = [train_cosmic_map[col] for col in train_rnadata.columns]
        test_rnadata.columns = [test_cosmic_map[col] for col in test_rnadata.columns]

        print("\nRNA data loaded:")
        print(f"Training samples: {train_rnadata.shape[1]}")
        print(f"Testing samples: {test_rnadata.shape[1]}")

        return train_rnadata, test_rnadata


if __name__ == '__main__':
    obj = GetData()
