import itertools
import mygene
import pandas as pd
import numpy as np
import obonet
import networkx as nx
import math
import sys
import util
from util import *
genes_file = "data/hotspot_gene_list_indexed.txt" # Path to the file with the genes
gene2id_mapping = load_mapping(genes_file)

mg = mygene.MyGeneInfo()
# Change gene symbol/name/alias to entrezgene id
genes_ids = mg.querymany(gene2id_mapping.keys(), scopes='symbol', species="human",fields="entrezgene",as_dataframe=True)
# genes_ids["query"] # has the genes symbols
genes_ids.reset_index(level=0, inplace=True) # remove row name (change to column)
genes_ids.dropna(subset=['entrezgene'], inplace=True) # remove genes with nans in entrezgene
genes_ids.drop_duplicates(subset=['query'], inplace=True) # some input query terms found dup hits, remove duplicates

# Get the annotations of the genes (split them in 2 because sometimes it crashes, Bad Gateway error)
# Define the number of genes and the split point
total_genes = len(genes_ids['entrezgene'])
split_point = math.ceil(total_genes / 2)

# Split the gene IDs into two halves 
first_half_genes = genes_ids['entrezgene'][:split_point]
second_half_genes = genes_ids['entrezgene'][split_point:]

# Query for the first half of the genes
first_half_annotations = mg.getgenes(first_half_genes, fields='symbol,go.BP.id', as_dataframe=True)
# Query for the second half of the genes
second_half_annotations = mg.getgenes(second_half_genes, fields='symbol,go.BP.id', as_dataframe=True)

# Merge the resulting dataframes
genes_annotations = pd.concat([first_half_annotations, second_half_annotations])
genes_annotations["symbol_ori"] = genes_ids["query"].values # change to the symbol used on the expression matrix (should be the same one, but just to make sure)

# Create go-gene dataframe of BP processes
gene_go = []
for gene,row in genes_annotations.iterrows():
    annotations = row["go.BP"] # can be modified to MF or CC
    if isinstance(annotations,list): # Ensures that further processing is only performed if the value is a list.
        terms = [item for sublist in annotations for item in sublist.values()] 
        pairs = itertools.product(terms,list([row["symbol_ori"]])) # Generates all possible combinations (Cartesian product) between the values in terms and a list containing the value from the "symbol" 
        result = [pair for pair in pairs if pair[0] != pair[1]]
        gene_go.append(result)
    elif isinstance(row["go.BP.id"],str): # if it only has 1 GO term, it is stored in go.BP.id
        gene_go.append([(row["go.BP.id"],row["symbol_ori"])])    
        
# flatten list
gene_go = np.array([item for sublist in gene_go for item in sublist])
gene_go = pd.DataFrame(gene_go).drop_duplicates()
# gene_go["type"] = ["gene"]*len(gene_go)


# 2. Download all the gene ontology network --------------

# Read the Gene Ontology ------
url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
full_graph = obonet.read_obo(url)
full_graph = full_graph.reverse() # change the direction of nodes
[n for n in full_graph.nodes if full_graph.in_degree(n) == 0] # graph contains the 3 roots (BP,MF,CC)

def remove_node(g, node):
    """
    Removes the node and connects edges (connects its parents with its childs)

        Parameters
        ----------
        g: A directed graph class that can store multiedges.

        node: str, name of term to remove

        Output
        ------
        g: The directed graph without the unwanted node

    """
    sources = [source for source, _ in g.in_edges(node)] # parents
    targets = [target for _, target in g.out_edges(node)] # children
    new_edges = itertools.product(sources, targets)
    new_edges = [(source, target) for source, target in new_edges if source != target] # remove self-loops
    g.add_edges_from(new_edges) # add new connections
    g.remove_node(node) # remove unwanted term

    return g

# 3.1. Remove non-annotated terms
all_nodes = set(full_graph.nodes()) # All nodes of the obo graph
keep_nodes = set(gene_go[0]) # Annotated nodes

# some terms are obsolete and have no parents, this can cause problems when creating the hierarchy (can't create all connections)
keep_nodes_aux = keep_nodes.copy()
for i in keep_nodes_aux:
    num_parents = len([source for source, _ in full_graph.in_edges(i)])
    if num_parents==0:
        # print(i)
        keep_nodes.remove(i)

keep_nodes.add("GO:0008150")

unwanted_nodes = all_nodes - keep_nodes

our_graph = full_graph.copy()

for term in unwanted_nodes:
    # print(term)
    remove_node(our_graph, term)

[n for n in our_graph.nodes if len(our_graph.in_edges(n)) == 0]  # roots HEREEE

# 3.2. Add the gene nodes
gene_go_list = list(gene_go.itertuples(index=False, name=None))
our_graph.add_edges_from(gene_go_list)

[n for n in our_graph.nodes if len(our_graph.in_edges(n)) == 0]

our_graph_copy = our_graph.copy()
# remove nodes that have genes but no parents (again, just to make sure), or genes whose annotations are not in the ontology (obsolete terms)
for node in our_graph_copy.nodes:
    if len(our_graph.in_edges(node)) == 0 and node != "GO:0008150":# if node has genes but no parents
        print(node)
        remove_node(our_graph, node)

del our_graph_copy # delete

[n for n in our_graph.nodes if len(our_graph.in_edges(n)) == 0]

# verify graph only has one root
import networkx.algorithms.components.connected as nxacc
uG = our_graph.to_undirected() # Returns an undirected representation of the digraph
connected_subG_list = list(nxacc.connected_components(uG)) #connected components
[n for n in our_graph.nodes if len(our_graph.in_edges(n)) == 0]  # roots

# Create go-gene/go dataframe and export to txt
edges = np.array(list(our_graph.edges())) # convert edges to a NumPy array
edges = np.unique(edges, axis=0) # drop duplicates
empty_col = np.empty((edges.shape[0], 1)) # add empty col
edges = np.hstack((edges, empty_col))
mask = np.char.startswith(edges[:,1], 'GO:') # create a mask for rows that start with 'GO:'
edges[:,2] = np.where(mask, 'default', 'gene') # create a new column based on the mask
edges_genes = edges[edges[:, 2] == 'gene']

pd.DataFrame(edges).to_csv("data/ontology_structure_gene_list_indexed.txt", sep='\t', index=False, header=False)
