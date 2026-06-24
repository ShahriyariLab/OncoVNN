import codecs
import os.path
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag

import scipy.stats as ss
from subword_nmt.apply_bpe import BPE


def spearman_corr(x, y):
    def _rank(t: torch.Tensor) -> torch.Tensor:
        order = t.argsort()
        ranks = torch.zeros_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(len(t), dtype=torch.float32, device=t.device)
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    return torch.sum(rx_c * ry_c) / (torch.norm(rx_c, 2) * torch.norm(ry_c, 2))

def load_mapping(mapping_file):

    mapping = {}

    file_handle = open(mapping_file)

    for line in file_handle:
        line = line.rstrip().split()
        mapping[line[1]] = int(line[0])

    file_handle.close()
    
    return mapping        

def load_ontology(file_name, gene2id_mapping):
    dG = nx.DiGraph()
    term_direct_gene_map = {}
    term_size_map = {}

    file_handle = open(file_name)

    gene_set = set()

    for line in file_handle:
        line = line.rstrip().split()
        
        if line[2] == 'default':
            dG.add_edge(line[0], line[1])
        else:
            if line[1] not in gene2id_mapping:
                continue

            if line[0] not in term_direct_gene_map:
                term_direct_gene_map[line[0]] = set()

            term_direct_gene_map[line[0]].add(gene2id_mapping[line[1]])

            gene_set.add(line[1])

    file_handle.close()

    print('There are %d genes' % len(gene_set))

    for term in dG.nodes():
        term_gene_set = set()

        if term in term_direct_gene_map:
            term_gene_set = term_direct_gene_map[term]

        deslist = nx.descendants(dG, term)

        for child in deslist:
            if child in term_direct_gene_map:
                term_gene_set = term_gene_set | term_direct_gene_map[child]

        # Log the term and its gene set size
        #print(f'Term: {term}, Gene Set Size: {len(term_gene_set)}')

        if len(term_gene_set) == 0:
            print('There is empty terms, please delete term: %s' % term)
            sys.exit(1)
        else:
            term_size_map[term] = len(term_gene_set)

    leaves = [n for n, d in dG.in_degree() if d == 0]

    uG = dG.to_undirected()
    connected_subG_list = list(nx.connected_components(uG))

    print('There are %d roots: %s' % (len(leaves), leaves[0]))
    print('There are %d terms' % len(dG.nodes()))
    print('There are %d connected components' % len(connected_subG_list))

    if len(leaves) > 1:
        print('There are more than 1 root of ontology. Please use only one root.')
        sys.exit(1)
    if len(connected_subG_list) > 1:
        print('There are more than connected components. Please connect them.')
        sys.exit(1)

    return dG, leaves[0], term_size_map, term_direct_gene_map

def remove_empty_terms(ontology_file, gene2id_mapping):
    term_direct_gene_map = {}
    empty_terms = set()

    # First pass: build the term_direct_gene_map and identify empty terms
    with open(ontology_file, 'r') as file:
        for line in file:
            parts = line.rstrip().split()
            if len(parts) < 3:
                continue
            if parts[2] != 'default' and parts[1] in gene2id_mapping:
                if parts[0] not in term_direct_gene_map:
                    term_direct_gene_map[parts[0]] = set()
                term_direct_gene_map[parts[0]].add(gene2id_mapping[parts[1]])
            elif parts[2] == 'default':
                if parts[0] not in term_direct_gene_map:
                    empty_terms.add(parts[0])
                if parts[1] not in term_direct_gene_map:
                    empty_terms.add(parts[1])

    # Remove terms from empty_terms if they have associated genes
    for term in term_direct_gene_map:
        if term in empty_terms:
            empty_terms.remove(term)

    # Second pass: write only non-empty terms and their edges
    with open(ontology_file, 'r') as file:
        lines = file.readlines()

    with open(ontology_file, 'w') as file:
        for line in lines:
            parts = line.rstrip().split()
            if len(parts) < 3:
                file.write(line)
                continue
            if parts[2] == 'default':
                if parts[0] not in empty_terms and parts[1] not in empty_terms:
                    file.write(line)
            elif parts[0] not in empty_terms:
                file.write(line)


def load_train_data(file_name, cell2id, drug2id):
    feature = []
    label = []

    with open(file_name, 'r') as fi:
        for line in fi:
            tokens = line.strip().split('\t')

            feature.append([cell2id[tokens[0]], drug2id[tokens[1]]])
            label.append([float(tokens[2])])

    return feature, label


def prepare_predict_data(test_file, cell2id_mapping_file, drug2id_mapping_file):

    # load mapping files
    cell2id_mapping = load_mapping(cell2id_mapping_file)
    drug2id_mapping = load_mapping(drug2id_mapping_file)

    test_feature, test_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping)

    print('Total number of cell lines = %d' % len(cell2id_mapping))
    print('Total number of drugs = %d' % len(drug2id_mapping))

    return (torch.Tensor(test_feature), torch.Tensor(test_label)), cell2id_mapping, drug2id_mapping


def prepare_train_data(train_file, test_file, cell2id_mapping_file, drug2id_mapping_file):

    # load mapping files
    cell2id_mapping = load_mapping(cell2id_mapping_file)
    drug2id_mapping = load_mapping(drug2id_mapping_file)

    train_feature, train_label = load_train_data(train_file, cell2id_mapping, drug2id_mapping)
    test_feature, test_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping)

    print('Total number of cell lines = %d' % len(cell2id_mapping))
    print('Total number of drugs = %d' % len(drug2id_mapping))

    return (torch.Tensor(train_feature), torch.FloatTensor(train_label), torch.Tensor(test_feature), torch.FloatTensor(test_label)), cell2id_mapping, drug2id_mapping



def build_input_vector(row_data, num_col, original_features):

    cuda_features = torch.zeros(len(row_data), num_col)

    for i in range(len(row_data)):
        data_ind = row_data[i]
        cuda_features.data[i] = original_features.data[data_ind]
   
    return cuda_features


def count_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Concat(nn.Module):
    """Squeeze wrapper for nn.Sequential."""
    def __init__(self, dim=-1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, input1, input2):
        return torch.cat([input1, input2], dim=self.dim)