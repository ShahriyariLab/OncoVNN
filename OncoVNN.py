import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
from prettytable import PrettyTable
import networkx as nx

from util import load_mapping, load_ontology
from data_loader import GetData
from data_encoding import DataEncoding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GENE2ID_FILE  = 'data/gene2id_mapping.txt'
ONTOLOGY_FILE = 'data/ontology_structure.txt'


class data_process_loader(data.Dataset):
    def __init__(self, list_IDs, labels, drug_df, rna_df):
        super().__init__()
        self.list_IDs = list_IDs
        self.labels = labels
        self.drug_df = drug_df

        # Convert RNA data to DataFrame if numpy array
        self.rna_df = pd.DataFrame(rna_df) if isinstance(rna_df, np.ndarray) else rna_df

        # Ensure RNA data has correct index
        if 'COSMIC_ID' in self.drug_df.columns:
            cosmic_ids = self.drug_df['COSMIC_ID'].astype(str)

            valid_indices = []
            for idx in self.list_IDs:
                if str(self.drug_df.iloc[idx]['COSMIC_ID']) in self.rna_df.index:
                    valid_indices.append(idx)

            self.list_IDs = valid_indices

            print(f"Data loader initialized with {len(self.list_IDs)} valid samples")
            print(f"RNA data shape: {self.rna_df.shape}")

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        idx = self.list_IDs[index]

        v_d = torch.tensor(self.drug_df.iloc[idx]['drug_encoding']).float()

        cosmic_id = str(self.drug_df.iloc[idx]['COSMIC_ID'])
        v_p = torch.tensor(self.rna_df.loc[cosmic_id].values).float()

        y = self.labels[idx]

        return v_d, v_p, y


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.input_dim_gene = 965
        self.hidden_dim_gene = 256
        self.mlp_hidden_dims_gene = [1024, 256, 64]

        dims = [self.input_dim_gene] + self.mlp_hidden_dims_gene + [self.hidden_dim_gene]
        self.predictor = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])

    def forward(self, v):
        v = v.float().to(device)
        for layer in self.predictor:
            v = F.relu(layer(v))
        return v

class VNN(nn.Module):
    def __init__(self):
        super().__init__()
        gene2id_mapping = load_mapping(GENE2ID_FILE)
        self.dG, self.root, term_size_map, self.term_direct_gene_map = load_ontology(
            ONTOLOGY_FILE, gene2id_mapping
        )

        self.num_hiddens_genotype = 6
        self.term_dim_map = {term: self.num_hiddens_genotype for term in term_size_map}

        # Sorted gene indices per term for reproducible sparse connectivity
        self.term_gene_indices = {
            term: sorted(genes)
            for term, genes in self.term_direct_gene_map.items()
        }

        all_ids = [i for ids in self.term_gene_indices.values() for i in ids]
        self._max_gene_id = max(all_ids) if all_ids else 0

        self.children_map = {term: list(self.dG.successors(term)) for term in self.dG.nodes()}

        # Bottom-up processing order: leaves first, root last
        self.term_order = list(reversed(list(nx.topological_sort(self.dG))))

        for term in self.dG.nodes():
            n_direct = len(self.term_gene_indices.get(term, []))
            n_children = sum(self.term_dim_map[c] for c in self.children_map[term])
            total_input = n_direct + n_children

            if n_direct > 0:
                self.add_module(f'{term}_direct_gene_layer', nn.Linear(n_direct, n_direct))

            if total_input > 0:
                self.add_module(f'{term}_GO_linear_layer',
                                nn.Linear(total_input, self.term_dim_map[term], bias=False))
                self.add_module(f'{term}_GO_batchnorm_layer',
                                nn.BatchNorm1d(self.term_dim_map[term]))

    @property
    def output_dim(self):
        return len(self.dG.nodes()) * self.num_hiddens_genotype

    def forward(self, gene_input):
        gene_input = gene_input.float().to(device)
        assert gene_input.shape[1] > self._max_gene_id, (
            f"Gene input has {gene_input.shape[1]} features but the ontology requires "
            f"at least {self._max_gene_id + 1}. Ensure RNA DataFrame columns are ordered "
            f"to match gene2id_mapping (column i must hold the gene with ID i)."
        )
        term_NN_out_map = {}

        for term in self.term_order:
            input_parts = []

            if term in self.term_gene_indices:
                indices = self.term_gene_indices[term]
                selected = gene_input[:, indices]
                input_parts.append(self._modules[f'{term}_direct_gene_layer'](selected))

            for child in self.children_map[term]:
                if child in term_NN_out_map:
                    input_parts.append(term_NN_out_map[child])

            if input_parts and f'{term}_GO_linear_layer' in self._modules:
                combined = torch.cat(input_parts, dim=1)
                x = self._modules[f'{term}_GO_linear_layer'](combined)
                x = self._modules[f'{term}_GO_batchnorm_layer'](x)
                x = F.relu(x)
                term_NN_out_map[term] = x

        return term_NN_out_map

class ModularClassifier(nn.Module):
    def __init__(self, input_dim, use_transformer=True, use_vnn=True, use_mlp=True):
        super().__init__()
        self.use_transformer = use_transformer
        self.use_vnn = use_vnn
        self.use_mlp = use_mlp

        hidden_dims = [1024, 512, 256]
        dims = [input_dim] + hidden_dims + [1]

        self.predictor = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.dropout = nn.Dropout(0.1)

    def forward(self, features):
        x = torch.cat(features, dim=1)

        for i, layer in enumerate(self.predictor):
            if i == len(self.predictor) - 1:
                x = layer(x)
            else:
                x = F.relu(layer(x))
                x = self.dropout(x)
        return x

class OncoVNN:
    def __init__(self, modeldir, use_transformer=True, use_vnn=True, use_mlp=True):
        """
        Initialize OncoVNN with configurable layers.

        Args:
            modeldir: Directory for model storage
            use_transformer: Whether to use transformer layer for drugs
            use_vnn: Whether to use VNN layer for gene ontology
            use_mlp: Whether to use MLP layer for gene expression
        """
        if not any([use_transformer, use_vnn, use_mlp]):
            raise ValueError("At least one layer must be enabled")

        self.use_transformer = use_transformer
        self.use_vnn = use_vnn
        self.use_mlp = use_mlp

        if use_vnn:
            self.model_gene = VNN().to(device)
            # Bottleneck projection: compress raw VNN output to 256 before fusion
            self.vnn_projection = nn.Linear(self.model_gene.output_dim, 256).to(device)
        if use_mlp:
            self.model_mlp = MLP().to(device)

        feature_dim = 0
        if use_transformer:
            feature_dim += 768
        if use_vnn:
            feature_dim += 256
        if use_mlp:
            feature_dim += self.model_mlp.hidden_dim_gene

        self.model = ModularClassifier(
            input_dim=feature_dim,
            use_transformer=use_transformer,
            use_vnn=use_vnn,
            use_mlp=use_mlp
        ).to(device)

        self.device = device
        self.modeldir = modeldir

        print("\nModel Configuration:")
        print(f"Using Transformer: {use_transformer}")
        print(f"Using VNN: {use_vnn}")
        print(f"Using MLP: {use_mlp}")
        print(f"Total feature dimension: {feature_dim}\n")

    def train(self, train_drug, train_rna, epochs=5):
        lr = 1e-5
        BATCH_SIZE = 64

        opt_params = list(self.model.parameters())
        if self.use_vnn:
            opt_params += list(self.model_gene.parameters())
            opt_params += list(self.vnn_projection.parameters())
        if self.use_mlp:
            opt_params += list(self.model_mlp.parameters())

        opt = torch.optim.Adam(opt_params, lr=lr)

        loader_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 0, 'drop_last': True}
        train_loader = data.DataLoader(
            data_process_loader(train_drug.index.values, train_drug.LN_IC50.values,
                                train_drug, train_rna), **loader_params)

        print("--- Training started ---")
        for epoch in range(epochs):
            self.model.train()
            if self.use_vnn:
                self.model_gene.train()
                self.vnn_projection.train()
            if self.use_mlp:
                self.model_mlp.train()

            total_loss = 0
            for v_d, v_p, label in train_loader:
                opt.zero_grad()
                v_d, v_p, label = v_d.to(device), v_p.to(device), label.to(device)

                features = []
                if self.use_transformer:
                    features.append(v_d)
                if self.use_vnn:
                    v_P_vnn_map = self.model_gene(v_p)
                    v_P_vnn = self.vnn_projection(torch.cat(list(v_P_vnn_map.values()), dim=1))
                    features.append(v_P_vnn)
                if self.use_mlp:
                    v_P_mlp = self.model_mlp(v_p)
                    features.append(v_P_mlp)

                output = self.model(features)
                loss = nn.MSELoss()(output.squeeze(-1), label.float())
                loss.backward()
                opt.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    def predict(self, test_drug, test_rna, return_labels=False):
        self.model.eval()
        if self.use_vnn:
            self.model_gene.eval()
            self.vnn_projection.eval()
        if self.use_mlp:
            self.model_mlp.eval()

        test_drug = test_drug.reset_index(drop=True)
        has_labels = 'LN_IC50' in test_drug.columns
        labels = test_drug['LN_IC50'].values if has_labels else np.zeros(len(test_drug))

        loader_params = {'batch_size': 64, 'shuffle': False, 'num_workers': 0, 'drop_last': False}
        test_loader = data.DataLoader(
            data_process_loader(
                list_IDs=list(range(len(test_drug))),
                labels=labels,
                drug_df=test_drug,
                rna_df=test_rna,
            ),
            **loader_params,
        )

        preds_list = []
        labels_list = []
        with torch.no_grad():
            for v_d, v_p, label in test_loader:
                v_d, v_p = v_d.to(self.device), v_p.to(self.device)

                features = []
                if self.use_transformer:
                    features.append(v_d)
                if self.use_vnn:
                    v_P_vnn_map = self.model_gene(v_p)
                    v_P_vnn = self.vnn_projection(torch.cat(list(v_P_vnn_map.values()), dim=1))
                    features.append(v_P_vnn)
                if self.use_mlp:
                    v_P_mlp = self.model_mlp(v_p)
                    features.append(v_P_mlp)

                output = self.model(features)
                preds_list.append(output.squeeze(-1).cpu().numpy())
                labels_list.append(label.numpy())

        preds = np.concatenate(preds_list, axis=0)
        if return_labels:
            return preds, np.concatenate(labels_list, axis=0)
        return preds

    def evaluate(self, test_drug, test_rna):
        if 'LN_IC50' not in test_drug.columns:
            print("[WARNING] 'LN_IC50' column not found. Skipping evaluation.")
            return {}

        y_pred, y_true = self.predict(test_drug, test_rna, return_labels=True)

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        pearson_corr, p_value = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        ci = concordance_index(y_true, y_pred)

        metrics_table = PrettyTable()
        metrics_table.field_names = ["Metric", "Value"]
        metrics_table.add_row(["MSE", f"{mse:.4f}"])
        metrics_table.add_row(["RMSE", f"{rmse:.4f}"])
        metrics_table.add_row(["Pearson", f"{pearson_corr:.4f} (p={p_value:.4f})"])
        metrics_table.add_row(["Spearman", f"{spearman_corr:.4f}"])
        metrics_table.add_row(["Concordance Index", f"{ci:.4f}"])
        print(metrics_table)

        return {
            'mse': mse,
            'rmse': rmse,
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'ci': ci
        }

    def save_model(self, path):
        state = {
            'config': {
                'use_transformer': self.use_transformer,
                'use_vnn': self.use_vnn,
                'use_mlp': self.use_mlp
            },
            'model_state': self.model.state_dict()
        }
        if self.use_vnn:
            state['vnn_state'] = self.model_gene.state_dict()
            state['vnn_projection_state'] = self.vnn_projection.state_dict()
        if self.use_mlp:
            state['mlp_state'] = self.model_mlp.state_dict()

        torch.save(state, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)

        if state['config'] != {
            'use_transformer': self.use_transformer,
            'use_vnn': self.use_vnn,
            'use_mlp': self.use_mlp
        }:
            raise ValueError("Model configuration mismatch")

        self.model.load_state_dict(state['model_state'])
        if self.use_vnn:
            self.model_gene.load_state_dict(state['vnn_state'])
            self.vnn_projection.load_state_dict(state['vnn_projection_state'])
        if self.use_mlp:
            self.model_mlp.load_state_dict(state['mlp_state'])
        print(f"Model loaded from {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and test OncoVNN model")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--predict', action='store_true', help='Run prediction')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--modeldir', type=str, default='./model_OncoVNN',
                        help='Directory to save/load the model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pretrained model for testing')
    parser.add_argument('--use_transformer', action='store_true',
                        help='Use transformer layer for drugs')
    parser.add_argument('--use_vnn', action='store_true',
                        help='Use VNN layer for gene ontology')
    parser.add_argument('--use_mlp', action='store_true',
                        help='Use MLP layer for gene expression')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)

    print("Loading data...")
    get_data = GetData()
    traindata, testdata = get_data.ByCancer(random_seed=args.random_seed)

    traindata = traindata.reset_index(drop=True)
    testdata = testdata.reset_index(drop=True)

    data_encoding = DataEncoding(vocab_dir='DeepTTC')
    traindata_encoded, train_rna, testdata_encoded, test_rna = data_encoding.encode(
        traindata=traindata,
        testdata=testdata
    )

    print(f"Training samples: {len(traindata_encoded)}")
    print(f"Test samples: {len(testdata_encoded)}")

    model = OncoVNN(
        modeldir=args.modeldir,
        use_transformer=args.use_transformer,
        use_vnn=args.use_vnn,
        use_mlp=args.use_mlp
    )

    if args.train:
        print("\nStarting training...")
        model.train(traindata_encoded, train_rna, epochs=args.epochs)
        os.makedirs(args.modeldir, exist_ok=True)
        model.save_model(os.path.join(args.modeldir, 'model.pt'))

    if args.test:
        if args.model_path:
            model.load_model(args.model_path)
        print("\nEvaluating on test set...")
        metrics = model.evaluate(testdata_encoded, test_rna)
        if metrics:
            metrics_path = os.path.join(args.modeldir, 'metrics.csv')
            pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
            print(f"Metrics saved to {metrics_path}")
