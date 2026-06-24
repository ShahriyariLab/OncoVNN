"""
attribution.py
==============
Loads a trained OncoVNN model and uses the GO hierarchy inside the VNN to
attribute drug response to genes and pathways (Hierarchical VNN GradCAM).

Attribution method: Hierarchical VNN GradCAM
─────────────────────────────────────────────
The VNN's hidden nodes ARE GO pathways.  Computing gradients at the raw
gene-expression input (standard IG) ignores this structure.  Instead we:

  1. Forward pass → capture each GO term's output activation h_t ∈ R[cells×6]
  2. Backward pass → compute ∂prediction/∂h_t for every term t
  3. Term relevance  = mean_cells(mean_dims( h_t × ∂pred/∂h_t ))
                     (GradCAM formula applied at term level)
  4. Gene importance = Σ_t∈terms(g)  term_relevance[t]
                                    × mean |direct_layer_weight[g]|
     where direct_layer_weight[g] is the learned weight connecting gene g
     into term t's linear layer.

This means a gene is important when it is a member of highly-activated,
drug-relevant pathways — not just when its own expression gradient is large.

Plot convention:
  Blue node  : gene_importance < 0  → sensitivity (pushes IC50 down)
  Red  node  : gene_importance > 0  → resistance  (pushes IC50 up)
  Node size  : proportional to |importance|
  Gray node  : STRING linker gene (connects ≥2 top genes)
  Gray edges : STRING PPI (score ≥ 400, cached)

Usage:
  python attribution.py
  python attribution.py --drugs "DASATINIB,LAPATINIB,TRAMETINIB"
  python attribution.py --topk 10 --score 700 --with_string
"""

import os, json, argparse, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import requests

warnings.filterwarnings('ignore')

from OncoVNN import OncoVNN, VNN
from data_loader import GetData
from data_encoding import DataEncoding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Device: {device}')

MODEL_PATH   = './model_OncoVNN/model.pt'   # path to trained model checkpoint
STRING_CACHE = './results/attribution/string_cache.json'
OUTDIR       = './results/attribution'
SPECIES      = 9606   # Homo sapiens

DEFAULT_DRUGS = [
    'DASATINIB', 'IBRUTINIB', 'CANERTINIB',
    'DACOMITINIB', 'LAPATINIB', 'PELITINIB',
    'POZIOTINIB', 'IDASANUTLIN', 'COBIMETINIB',
    'TRAMETINIB', 'VINBLASTINE', 'VINCRISTINE',
]


# ── Gene name utilities ───────────────────────────────────────────────────────

def load_gene_names():
    """Return list of 965 gene names ordered by gene2id index."""
    gene2id = {}
    with open('data/gene2id_mapping.txt') as f:
        for line in f:
            parts = line.rstrip().split()
            gene2id[parts[1]] = int(parts[0])
    idx_to_gene = {v: k for k, v in gene2id.items()}
    return [idx_to_gene[i] for i in range(len(idx_to_gene))]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_trained_model(path):
    model = OncoVNN(
        modeldir=os.path.dirname(path),
        use_transformer=True,
        use_vnn=True,
        use_mlp=True,
    )
    model.load_model(path)
    model.model.eval()
    model.model_gene.eval()
    model.vnn_projection.eval()
    model.model_mlp.eval()
    return model


# ── VNN Hierarchical GradCAM attribution ─────────────────────────────────────

def compute_vnn_gene_attribution(model, v_d, expr_tensor):
    """
    Compute gene importance scores by applying GradCAM at every GO term node
    in the VNN, then propagating term relevance back to genes via learned weights.

    Steps
    -----
    1. Forward pass with hooks to capture every term's output h_t [n_cells, 6].
    2. Backward pass to obtain ∂prediction/∂h_t for each term.
    3. term_relevance[t] = mean_cells( mean_dims( h_t × ∂pred/∂h_t ) )
       — positive = term activations help predict high IC50 (resistance)
       — negative = term activations help predict low  IC50 (sensitivity)
    4. gene_importance[g] = Σ_{t ∈ terms(g)}
           term_relevance[t] × mean|direct_layer_weight of g in t|
       so a gene inherits importance from every pathway it belongs to,
       weighted by the strength of its learned connection.

    Parameters
    ----------
    v_d         : [n_cells, 768]  drug embedding (fixed, no grad needed)
    expr_tensor : [n_cells, 965]  gene expression for this drug's cell lines

    Returns
    -------
    gene_scores : ndarray [965]  signed importance (positive=resistance,
                                                    negative=sensitivity)
    term_relevance : dict  {term_id: float}  pathway-level relevance
    """
    vnn = model.model_gene
    n_genes = expr_tensor.shape[1]

    v_d         = v_d.to(device)
    expr_tensor = expr_tensor.to(device)

    # ── Step 1: forward with activation hooks ─────────────────────────────
    activations = {}
    hooks = []

    def make_hook(term_name):
        def hook(_module, _inp, out):
            activations[term_name] = out
        return hook

    for term in vnn.dG.nodes():
        mod_name = f'{term}_GO_linear_layer'
        if mod_name in vnn._modules:
            h = vnn._modules[mod_name].register_forward_hook(make_hook(term))
            hooks.append(h)

    vnn.eval(); model.vnn_projection.eval()
    model.model_mlp.eval(); model.model.eval()

    expr_in = expr_tensor.detach().requires_grad_(False)
    vnn_map = vnn(expr_in)
    vnn_cat = torch.cat(list(vnn_map.values()), dim=1)
    v_vnn   = model.vnn_projection(vnn_cat)
    v_mlp   = model.model_mlp(expr_in)
    pred    = model.model([v_d, v_vnn, v_mlp])

    for h in hooks:
        h.remove()

    # ── Step 2: backward to get gradients w.r.t. each term's output ───────
    gradients = {}

    def make_grad_hook(term_name):
        def grad_hook(grad):
            gradients[term_name] = grad.detach()
        return grad_hook

    grad_hooks = []
    for term, act in activations.items():
        act.retain_grad()
        gh = act.register_hook(make_grad_hook(term))
        grad_hooks.append(gh)

    pred.sum().backward()

    for gh in grad_hooks:
        gh.remove()

    # ── Step 3: term relevance = mean_cells(mean_dims(h_t × ∂pred/∂h_t)) ─
    term_relevance = {}
    for term, act in activations.items():
        if term not in gradients:
            continue
        rel = (act.detach() * gradients[term]).mean().item()
        term_relevance[term] = rel

    # ── Step 4: propagate term relevance → gene importance ────────────────
    gene_scores = np.zeros(n_genes)

    for term, rel in term_relevance.items():
        if term not in vnn.term_gene_indices:
            continue
        gene_indices = vnn.term_gene_indices[term]

        mod_name = f'{term}_direct_gene_layer'
        if mod_name in vnn._modules:
            W = vnn._modules[mod_name].weight.detach().cpu().numpy()
            gene_weights = np.abs(W).mean(axis=0)
        else:
            gene_weights = np.ones(len(gene_indices))

        for local_idx, global_idx in enumerate(gene_indices):
            w = gene_weights[local_idx] if local_idx < len(gene_weights) else 1.0
            gene_scores[global_idx] += rel * w

    return gene_scores, term_relevance


# ── STRING API ────────────────────────────────────────────────────────────────

def _load_string_cache():
    if os.path.exists(STRING_CACHE):
        with open(STRING_CACHE) as f:
            return json.load(f)
    return {}


def _save_string_cache(cache):
    os.makedirs(os.path.dirname(STRING_CACHE), exist_ok=True)
    with open(STRING_CACHE, 'w') as f:
        json.dump(cache, f)


def query_string_interactions(genes, score_threshold=400):
    """Return list of (gene_a, gene_b, score) for interactions among `genes`."""
    cache = _load_string_cache()
    cache_key = '|'.join(sorted(genes)) + f'@{score_threshold}'

    if cache_key in cache:
        return cache[cache_key]

    print(f'  [STRING] Querying {len(genes)} genes ...')
    try:
        resp = requests.post(
            'https://string-db.org/api/json/get_string_ids',
            data={
                'identifiers': '\r'.join(genes),
                'species': SPECIES,
                'caller_identity': 'oncovnn_attribution',
            },
            timeout=30,
        )
        if resp.status_code != 200 or not resp.json():
            cache[cache_key] = []
            _save_string_cache(cache)
            return []

        resolved = resp.json()
        id_map = {}
        for r in resolved:
            pref = r.get('preferredName', '').upper()
            if pref in [g.upper() for g in genes]:
                id_map[pref] = r['stringId']

        string_ids = list(set(id_map.values()))
        if not string_ids:
            cache[cache_key] = []
            _save_string_cache(cache)
            return []

        resp2 = requests.post(
            'https://string-db.org/api/json/network',
            data={
                'identifiers': '\r'.join(string_ids),
                'species': SPECIES,
                'required_score': score_threshold,
                'add_nodes': 10,
                'caller_identity': 'oncovnn_attribution',
            },
            timeout=30,
        )
        if resp2.status_code != 200:
            cache[cache_key] = []
            _save_string_cache(cache)
            return []

        edges = []
        for e in resp2.json():
            a = e.get('preferredName_A', '').upper()
            b = e.get('preferredName_B', '').upper()
            s = float(e.get('score', 0))
            if a and b and a != b:
                edges.append((a, b, s))

        time.sleep(1)
        cache[cache_key] = edges
        _save_string_cache(cache)
        return edges

    except Exception as exc:
        print(f'  [STRING] Error: {exc}')
        cache[cache_key] = []
        _save_string_cache(cache)
        return []


# ── Network construction ──────────────────────────────────────────────────────

def build_drug_network(top_genes, gene_scores, string_edges, n_linkers=15):
    top_set = set(g.upper() for g in top_genes)

    linker_count = {}
    all_edges = []
    for a, b, s in string_edges:
        a, b = a.upper(), b.upper()
        all_edges.append((a, b, s))
        if a not in top_set and b in top_set:
            linker_count[a] = linker_count.get(a, 0) + 1
        if b not in top_set and a in top_set:
            linker_count[b] = linker_count.get(b, 0) + 1

    linker_genes = sorted(
        [g for g, cnt in linker_count.items() if cnt >= 2],
        key=lambda g: -linker_count[g]
    )[:n_linkers]
    keep = top_set | set(linker_genes)

    G = nx.Graph()
    for gene in top_genes:
        G.add_node(gene.upper(), kind='top')
    for gene in linker_genes:
        G.add_node(gene.upper(), kind='linker')
    for a, b, s in all_edges:
        if a in keep and b in keep:
            G.add_edge(a, b, weight=s)

    return G, linker_genes


# ── Plot utilities ────────────────────────────────────────────────────────────

BLUE = '#4472C4'
RED  = '#C0504D'
GRAY = '#BFBFBF'

DRUG_CLASS = {
    'GEFITINIB':    ('EGFR inhibitor',          '#E87722'),
    'AFATINIB':     ('EGFR inhibitor',          '#E87722'),
    'VINBLASTINE':  ('Microtubule poison',       '#2E9E5F'),
    'DOCETAXEL':    ('Microtubule poison',       '#2E9E5F'),
    'CISPLATIN':    ('DNA-damage agent',         '#C0392B'),
    'CYTARABINE':   ('DNA-damage agent',         '#C0392B'),
    'OLAPARIB':     ('Checkpoint inhibitor',     '#8E44AD'),
    'AZD7762':      ('Checkpoint inhibitor',     '#8E44AD'),
    'KU-55933':     ('Checkpoint inhibitor',     '#8E44AD'),
    'SB216763':     ('Kinase inhibitor',         '#2471A3'),
    'IMATINIB':     ('Kinase inhibitor',         '#2471A3'),
    'DASATINIB':    ('Kinase inhibitor',         '#2471A3'),
    'NILOTINIB':    ('Kinase inhibitor',         '#2471A3'),
    'AXITINIB':     ('VEGFR inhibitor',          '#117A65'),
    'NAVITOCLAX':   ('BCL-2 inhibitor',          '#1A5276'),
    'VORINOSTAT':   ('HDAC inhibitor',           '#7E5109'),
    'CAMPTOTHECIN': ('Topo I inhibitor',         '#922B21'),
    'STAUROSPORINE':('Pan-kinase inhibitor',     '#1B4F72'),
}

BARPLOT_DRUGS = [
    'GEFITINIB',  'AFATINIB',   'VINBLASTINE', 'DOCETAXEL',
    'CISPLATIN',  'CYTARABINE', 'OLAPARIB',    'SB216763',
]

BARPLOT_CLASS = {
    'GEFITINIB':  'EGFR inhibitor',
    'AFATINIB':   'EGFR inhibitor',
    'VINBLASTINE':'Microtubule poison',
    'DOCETAXEL':  'Microtubule poison',
    'CISPLATIN':  'DNA-damage agent',
    'CYTARABINE': 'DNA-damage agent',
    'OLAPARIB':   'Checkpoint inhibitor',
    'SB216763':   'Kinase inhibitor\n(pathway outlier)',
}


def _drug_row_colors(drug_names):
    colors = [DRUG_CLASS.get(d, ('Unknown', '#AAAAAA'))[1] for d in drug_names]
    return pd.DataFrame({'Drug class': colors}, index=drug_names)


def _load_go_names():
    try:
        from goatools.obo_parser import GODag
        for obo in ['go-basic.obo', 'go.obo']:
            if os.path.exists(obo):
                dag = GODag(obo, optional_attrs=[])
                return {t: dag[t].name for t in dag}
    except Exception:
        pass
    try:
        import obonet
        for obo in ['go-basic.obo', 'go.obo']:
            if os.path.exists(obo):
                g = obonet.read_obo(obo)
                return {n: d.get('name', n) for n, d in g.nodes(data=True)}
    except Exception:
        pass
    return {}


def _short_go_label(go_id, go_names, maxlen=32):
    name = go_names.get(go_id, go_id)
    if len(name) > maxlen:
        name = name[:maxlen - 1] + '…'
    return name


def _class_legend_patches():
    seen = {}
    for label, color in DRUG_CLASS.values():
        seen[label] = color
    import matplotlib.patches as mp
    return [mp.Patch(color=c, label=l) for l, c in seen.items()]


def plot_drug_network(ax, drug_name, G, top_genes, linker_genes, gene_scores):
    top_set    = set(g.upper() for g in top_genes)
    linker_set = set(g.upper() for g in linker_genes)

    connected_nodes = {n for n in G.nodes() if G.degree(n) > 0}
    isolated_top    = sorted(top_set - connected_nodes)
    H = G.subgraph(connected_nodes).copy()

    ax.set_title(drug_name, fontweight='bold', fontsize=9, pad=4)

    if H.number_of_nodes() == 0:
        ax.text(0.5, 0.55, 'No STRING\ninteractions found',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=7, color='#999999', style='italic')
        if isolated_top:
            ax.text(0.5, 0.25, '\n'.join(
                        ', '.join(isolated_top[i:i+4])
                        for i in range(0, min(len(isolated_top), 8), 4)
                    ),
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=5, color='#AAAAAA')
        ax.axis('off')
        return

    try:
        pos = nx.kamada_kawai_layout(H)
    except Exception:
        pos = nx.spring_layout(H, seed=42, k=2.0, iterations=120)

    widths = [max(0.5, H[u][v].get('weight', 400) / 500) for u, v in H.edges()]
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.30,
                           edge_color='#AAAAAA', width=widths)

    linker_nodes = [n for n in H.nodes() if n in linker_set]
    if linker_nodes:
        nx.draw_networkx_nodes(H, pos, nodelist=linker_nodes, ax=ax,
                               node_color=GRAY, node_size=120, alpha=0.7)
        nx.draw_networkx_labels(H, pos,
                                labels={n: n for n in linker_nodes},
                                ax=ax, font_size=4.5, font_color='#666666')

    max_abs = max(abs(gene_scores.get(g, 1e-9)) for g in top_set) or 1.0
    for gene in top_set:
        if gene not in H.nodes():
            continue
        score = gene_scores.get(gene, 0.0)
        color = BLUE if score < 0 else RED
        size  = 300 + 1600 * (abs(score) / max_abs)
        nx.draw_networkx_nodes(H, pos, nodelist=[gene], ax=ax,
                               node_color=color, node_size=size,
                               linewidths=0.9, edgecolors='white')

    top_labels = {n: n for n in top_set if n in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels=top_labels, ax=ax,
                            font_size=6, font_weight='bold', font_color='black')

    if isolated_top:
        ax.text(0.5, 0.01,
                'Unconnected: ' + ', '.join(isolated_top[:7])
                + ('…' if len(isolated_top) > 7 else ''),
                ha='center', va='bottom', transform=ax.transAxes,
                fontsize=4, color='#AAAAAA', style='italic')

    ax.axis('off')


def plot_attribution_heatmap(ig_results, gene_names, outdir, topk_per_drug=10):
    try:
        import seaborn as sns
    except ImportError:
        print('[WARN] seaborn not installed — skipping heatmap. pip install seaborn')
        return

    drug_names = [d for d in BARPLOT_DRUGS if d in ig_results]
    if not drug_names:
        drug_names = list(ig_results.keys())

    matrix  = np.vstack([ig_results[d] for d in drug_names])
    df_full = pd.DataFrame(matrix, index=drug_names, columns=gene_names)

    selected = set()
    for drug in drug_names:
        top = df_full.loc[drug].abs().nlargest(topk_per_drug).index.tolist()
        selected.update(top)
    selected = sorted(selected)
    df_sub = df_full[selected]

    scale   = df_sub.abs().quantile(0.95, axis=1).clip(lower=1e-9)
    df_norm = df_sub.div(scale, axis=0)
    vmax    = float(np.percentile(np.abs(df_norm.values), 98))
    vmax    = max(vmax, 0.1)

    n_drugs, n_genes = df_norm.shape

    fig_w = max(20, n_genes * 0.60)
    fig_h = max(10, n_drugs * 0.80)

    row_colors = _drug_row_colors(drug_names)
    mean_dir   = df_norm.mean(axis=0)
    col_colors = pd.DataFrame(
        {'Mean direction': [RED if s > 0 else BLUE for s in mean_dir]},
        index=df_norm.columns,
    )

    g = sns.clustermap(
        df_norm,
        cmap='RdBu_r',
        center=0, vmin=-vmax, vmax=vmax,
        figsize=(fig_w, fig_h),
        row_cluster=True, col_cluster=True,
        method='ward', metric='euclidean',
        yticklabels=True, xticklabels=True,
        linewidths=0.5, linecolor='#e0e0e0',
        row_colors=row_colors, col_colors=col_colors,
        cbar_pos=(0.01, 0.80, 0.018, 0.12),
        cbar_kws={'label': 'Normalised score', 'ticks': [-vmax, 0, vmax]},
        dendrogram_ratio=(0.10, 0.06),
        colors_ratio=0.022, tree_kws={'linewidths': 1.2},
    )

    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(),
        fontsize=11, fontweight='bold', rotation=55, ha='right', va='top',
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(),
        fontsize=13, fontweight='bold', rotation=0,
    )
    g.ax_heatmap.set_xlabel('Gene', labelpad=8, fontsize=14, fontweight='bold')
    g.ax_heatmap.set_ylabel('Drug', labelpad=8, fontsize=14, fontweight='bold')

    g.cax.tick_params(labelsize=10)
    g.cax.text(0.5,  1.08, '+ Resistance', ha='center', va='bottom',
               fontsize=9, fontweight='bold', color=RED, transform=g.cax.transAxes)
    g.cax.text(0.5, -0.12, '− Sensitivity', ha='center', va='top',
               fontsize=9, fontweight='bold', color=BLUE, transform=g.cax.transAxes)

    class_patches = _class_legend_patches()
    dir_patches   = [
        mpatches.Patch(color=RED,  label='Resistance driver (col avg +)'),
        mpatches.Patch(color=BLUE, label='Sensitivity driver (col avg −)'),
    ]
    g.ax_heatmap.legend(
        handles=class_patches + dir_patches,
        loc='upper left', bbox_to_anchor=(1.18, 1.0),
        fontsize=10, frameon=True, framealpha=0.9,
        title='Drug class  /  Gene direction', title_fontsize=10,
    )

    g.figure.suptitle(
        'Drug–Gene Attribution Heatmap  (VNN GradCAM  ·  Ward Linkage)',
        y=1.02, fontsize=16, fontweight='bold',
    )

    path = os.path.join(outdir, 'drug_gene_heatmap.png')
    g.figure.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[Output] Attribution heatmap    → {path}')


def plot_ranked_barplots(ig_results, gene_names, outdir, topk=10):
    ordered = [d for d in BARPLOT_DRUGS if d in ig_results]
    if not ordered:
        ordered = list(ig_results.keys())

    ncols = 4
    nrows = 2
    panel_w, panel_h = 7.0, 5.2

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows),
                             facecolor='white')
    axes = np.array(axes).flatten()

    for ax_idx, drug in enumerate(ordered):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        ig = ig_results[drug]

        top_idx   = np.argsort(np.abs(ig))[::-1][:topk]
        top_genes = [gene_names[i] for i in top_idx]
        scores    = [float(ig[i])   for i in top_idx]

        order     = np.argsort(scores)
        top_genes = [top_genes[i] for i in order]
        scores    = [scores[i]    for i in order]

        colors = [BLUE if s < 0 else RED for s in scores]
        y_pos  = list(range(len(scores)))

        ax.barh(y_pos, scores, color=colors,
                edgecolor='white', linewidth=0.4, height=0.72)

        for k in range(0, len(scores), 2):
            ax.axhspan(k - 0.5, k + 0.5, color='#f5f5f5', zorder=0)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_genes, fontsize=11, fontweight='bold')
        ax.axvline(0, color='#444444', linewidth=1.0)

        class_lbl = BARPLOT_CLASS.get(drug, '')
        ax.set_title(f'{drug}\n{class_lbl}', fontweight='bold', fontsize=12,
                     pad=5, color='#1a1a1a', linespacing=1.4)

        ax.tick_params(axis='x', labelsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        xmax = max(abs(s) for s in scores) * 1.20 or 1.0
        ax.set_xlim(-xmax, xmax)
        ax.set_xlabel('Attribution score', fontsize=9)

    for ax_idx in range(len(ordered), len(axes)):
        axes[ax_idx].set_visible(False)

    for row in range(nrows):
        axes[row * ncols + 1].spines['right'].set_visible(True)
        axes[row * ncols + 1].spines['right'].set_linewidth(1.2)
        axes[row * ncols + 1].spines['right'].set_color('#aaaaaa')

    legend_patches = [
        mpatches.Patch(color=BLUE, label='Sensitivity driver  (score < 0)'),
        mpatches.Patch(color=RED,  label='Resistance driver  (score > 0)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=2,
               fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f'Top-{topk} Predictive Genes — Representative Drug Classes  (VNN GradCAM)',
        fontsize=16, fontweight='bold', y=1.01,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.99])

    path = os.path.join(outdir, 'drug_gene_barplots.png')
    plt.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[Output] Ranked bar charts      → {path}')


def plot_pathway_heatmap(term_results, outdir, topk_terms=6):
    try:
        import seaborn as sns
    except ImportError:
        print('[WARN] seaborn not installed — skipping pathway heatmap.')
        return

    drug_names = sorted(term_results.keys())
    if not drug_names:
        return

    go_names = _load_go_names()
    if go_names:
        print(f'[GO] Loaded {len(go_names):,} GO term names.')
    else:
        print('[GO] No OBO file found — using GO IDs as labels.')

    selected_terms = set()
    for drug in drug_names:
        top = sorted(term_results[drug].items(),
                     key=lambda x: abs(x[1]), reverse=True)[:topk_terms]
        selected_terms.update(t for t, _ in top)
    selected_terms = sorted(selected_terms)

    matrix = []
    for drug in drug_names:
        row = [term_results[drug].get(t, 0.0) for t in selected_terms]
        matrix.append(row)

    readable_labels = [_short_go_label(t, go_names) for t in selected_terms]
    df = pd.DataFrame(matrix, index=drug_names, columns=readable_labels)

    scale   = df.abs().quantile(0.95, axis=1).clip(lower=1e-9)
    df_norm = df.div(scale, axis=0)
    vmax    = float(np.percentile(np.abs(df_norm.values), 98))
    vmax    = max(vmax, 0.1)

    df_plot  = df_norm.T
    n_terms, n_drugs = df_plot.shape

    fig_w = max(16, n_drugs * 1.20)
    fig_h = max(10, n_terms * 0.65)

    g = sns.clustermap(
        df_plot,
        cmap='RdBu_r',
        center=0, vmin=-vmax, vmax=vmax,
        figsize=(fig_w, fig_h),
        row_cluster=True, col_cluster=True,
        method='ward', metric='euclidean',
        yticklabels=True, xticklabels=True,
        linewidths=0.6, linecolor='#d8d8d8',
        cbar_pos=(0.01, 0.08, 0.020, 0.12),
        cbar_kws={'label': 'Normalised relevance', 'ticks': [-vmax, 0, vmax]},
        dendrogram_ratio=(0.08, 0.10), tree_kws={'linewidths': 1.0},
    )

    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(),
        fontsize=12, fontweight='bold', rotation=0,
    )
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(),
        fontsize=13, fontweight='bold', rotation=40, ha='right', va='top',
    )
    g.ax_heatmap.set_ylabel('GO Biological-Process Term', labelpad=10,
                            fontsize=14, fontweight='bold')
    g.ax_heatmap.set_xlabel('Drug', labelpad=10, fontsize=14, fontweight='bold')

    g.cax.tick_params(labelsize=10)
    g.cax.text(0.5,  1.10, '+ Resistance', ha='center', va='bottom',
               fontsize=9, fontweight='bold', color=RED, transform=g.cax.transAxes)
    g.cax.text(0.5, -0.14, '− Sensitivity', ha='center', va='top',
               fontsize=9, fontweight='bold', color=BLUE, transform=g.cax.transAxes)

    g.figure.suptitle(
        'Drug–Pathway Attribution Heatmap  (VNN GradCAM  ·  Ward Linkage)\n'
        'Rows = GO Biological-Process terms  ·  Columns = Drugs',
        y=1.02, fontsize=16, fontweight='bold',
    )

    path = os.path.join(outdir, 'drug_pathway_heatmap.png')
    g.figure.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[Output] Pathway heatmap        → {path}')


def plot_drug_pca(ig_results, gene_names, outdir):
    from sklearn.decomposition import PCA

    drug_names = list(ig_results.keys())
    matrix     = np.vstack([ig_results[d] for d in drug_names])

    pca    = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(matrix)
    var    = pca.explained_variance_ratio_

    _, ax = plt.subplots(figsize=(14, 11), facecolor='white')

    xc, yc = coords[:, 0], coords[:, 1]
    xspan  = xc.max() - xc.min() or 1.0
    yspan  = yc.max() - yc.min() or 1.0
    xpad, ypad = xspan * 0.42, yspan * 0.42
    xlim = (xc.min() - xpad, xc.max() + xpad)
    ylim = (yc.min() - ypad, yc.max() + ypad)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    loadings  = pca.components_.T
    load_mag  = np.linalg.norm(loadings, axis=1)
    top_load  = np.argsort(load_mag)[::-1][:12]
    max_reach = min(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 0.44
    scale     = max_reach / (np.abs(loadings[top_load]).max() or 1)

    gene_texts = []
    for idx in top_load:
        lx, ly = loadings[idx] * scale
        gene   = gene_names[idx]
        color  = RED if loadings[idx, 0] > 0 else BLUE
        ax.annotate('', xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=2.0, shrinkA=0, shrinkB=0),
                    zorder=3)
        mag = np.hypot(lx, ly) or 1e-9
        ox  = lx / mag * 0.028 * (xlim[1] - xlim[0])
        oy  = ly / mag * 0.028 * (ylim[1] - ylim[0])
        ha  = 'left'   if lx >= 0 else 'right'
        va  = 'bottom' if ly >= 0 else 'top'
        t = ax.text(lx + ox, ly + oy, gene,
                    fontsize=10, fontweight='bold', color=color,
                    ha=ha, va=va, zorder=5,
                    bbox=dict(boxstyle='round,pad=0.18', fc='white',
                              ec='none', alpha=0.82))
        gene_texts.append(t)

    for i in range(len(drug_names)):
        x, y = coords[i]
        ax.scatter(x, y, s=100, color="#000000", zorder=6,
                   edgecolors='white', linewidths=1.2)

    try:
        from adjustText import adjust_text
        drug_texts = []
        for i, drug in enumerate(drug_names):
            x, y = coords[i]
            t = ax.text(x, y, drug, fontsize=11, fontweight='bold',
                        color='#111111', zorder=7)
            drug_texts.append(t)
        adjust_text(
            drug_texts + gene_texts, ax=ax,
            x=coords[:, 0], y=coords[:, 1],
            expand_points=(2.4, 2.4), expand_text=(1.8, 1.8),
            force_points=(0.5, 0.5),
            arrowprops=dict(arrowstyle='-', color='#999999', lw=0.7),
        )
    except ImportError:
        x_med = float(np.median(xc))
        y_med = float(np.median(yc))
        for i, drug in enumerate(drug_names):
            x, y = coords[i]
            sign_x = 1  if x >= x_med else -1
            sign_y = 1  if y >= y_med else -1
            ha     = 'left'   if sign_x == 1 else 'right'
            va     = 'bottom' if sign_y == 1 else 'top'
            ax.annotate(drug, (x, y),
                        fontsize=11, fontweight='bold', color='#111111',
                        xytext=(9 * sign_x, 6 * sign_y),
                        textcoords='offset points',
                        ha=ha, va=va, zorder=7)

    ax.axhline(0, color='#C8C8C8', linewidth=0.9, zorder=1)
    ax.axvline(0, color='#C8C8C8', linewidth=0.9, zorder=1)
    ax.set_xlabel(f'PC1  ({var[0]*100:.1f}% variance)',
                  fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel(f'PC2  ({var[1]*100:.1f}% variance)',
                  fontsize=13, fontweight='bold', labelpad=10)
    ax.tick_params(labelsize=11)
    ax.set_title(
        'Drug Similarity by Gene Attribution Profile  (PCA Biplot)\n'
        'Proximity = shared gene dependency  |  Arrows = top gene loadings',
        fontsize=14, fontweight='bold', pad=14,
    )
    ax.spines[['top', 'right']].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=RED,  label='Gene loading -> positive PC1'),
        mpatches.Patch(color=BLUE, label='Gene loading -> negative PC1'),
    ]
    ax.legend(handles=legend_patches, fontsize=11, frameon=True,
              framealpha=0.90, edgecolor='#cccccc', loc='lower right')

    path = os.path.join(outdir, 'drug_attribution_pca.png')
    plt.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[Output] Drug PCA biplot        → {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drugs',    type=str,  default=None,
                        help='Comma-separated drug names (default: all in dataset)')
    parser.add_argument('--topk',     type=int,  default=10,
                        help='Top-K genes per drug (default 10)')
    parser.add_argument('--score',    type=int,  default=700,
                        help='STRING minimum interaction score (default 700 = high confidence)')
    parser.add_argument('--n_linkers', type=int,  default=15)
    parser.add_argument('--outdir',    type=str,  default=OUTDIR)
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                        help='Path to trained OncoVNN model checkpoint')
    parser.add_argument('--seed',      type=int,  default=42)
    parser.add_argument('--with_string', action='store_true',
                        help='Also query STRING and produce network plot (slow, requires internet)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    target_drugs = (
        [d.strip().upper() for d in args.drugs.split(',')]
        if args.drugs else None
    )

    # ── Load data ──────────────────────────────────────────────────────────
    print('[Data] Loading GDSC data...')
    get_data   = GetData()
    data_enc   = DataEncoding(vocab_dir='DeepTTC')
    traindata, testdata = get_data.ByCancer(random_seed=args.seed)
    traindata = traindata.reset_index(drop=True)
    testdata  = testdata.reset_index(drop=True)
    traindata_enc, train_rna, testdata_enc, test_rna = data_enc.encode(
        traindata=traindata, testdata=testdata
    )

    all_drug_df = pd.concat([traindata_enc, testdata_enc], ignore_index=True)
    all_rna_df  = pd.concat([train_rna,     test_rna    ], axis=0)
    all_rna_df  = all_rna_df[~all_rna_df.index.duplicated(keep='first')]

    gene_names = load_gene_names()

    # ── Load model ─────────────────────────────────────────────────────────
    print(f'[Model] Loading from {args.model_path}...')
    model = load_trained_model(args.model_path)

    # ── Drug ID → name mapping ─────────────────────────────────────────────
    drug_info_path = 'GDSC_data/Drug_listTue_Aug10_2021.csv'
    drug_info = pd.read_csv(drug_info_path)
    drugid_to_name = dict(zip(drug_info['drug_id'].astype(int),
                              drug_info['Name'].str.upper()))
    all_drug_df['DRUG_NAME'] = (all_drug_df['DRUG_ID']
                                .astype(int)
                                .map(drugid_to_name)
                                .fillna(all_drug_df['DRUG_ID'].astype(str)))

    available_names = all_drug_df['DRUG_NAME'].unique().tolist()
    print(f'[Data] {len(available_names)} unique drugs in dataset')

    if target_drugs:
        available = [d for d in target_drugs if d in available_names]
        missing   = [d for d in target_drugs if d not in available_names]
        if missing:
            print(f'[WARN] Requested drugs not in dataset: {missing}')
        if not available:
            print('[INFO] Falling back to all available drugs.')
            available = available_names
    else:
        available = available_names

    # ── Per-drug VNN GradCAM attribution ──────────────────────────────────
    grad_results = {}
    term_results = {}
    print(f'\n[Attribution] VNN GradCAM for {len(available)} drugs ...')

    for drug in available:
        rows = all_drug_df[all_drug_df['DRUG_NAME'] == drug].reset_index(drop=True)

        v_d_np = np.stack(rows['drug_encoding'].values)
        v_d    = torch.tensor(v_d_np, dtype=torch.float32).to(device)

        cosmic_ids = rows['COSMIC_ID'].astype(str).tolist()
        valid      = [c for c in cosmic_ids if c in all_rna_df.index]
        if not valid:
            print(f'  [Skip] {drug}: no matching RNA rows')
            continue

        expr_np = all_rna_df.loc[valid].values.astype(np.float32)
        expr_t  = torch.tensor(expr_np, dtype=torch.float32)
        v_d_cells = v_d[:len(valid)]

        gene_scores, term_rel = compute_vnn_gene_attribution(model, v_d_cells, expr_t)
        grad_results[drug] = gene_scores
        term_results[drug] = term_rel
        top5 = [gene_names[i]
                for i in np.argsort(np.abs(gene_scores))[::-1][:5]]
        print(f'  {drug:<25s}  top genes: {", ".join(top5)}')

    ig_results = grad_results

    # ── Save full attribution matrix ───────────────────────────────────────
    ig_df = pd.DataFrame(
        {drug: ig for drug, ig in ig_results.items()},
        index=gene_names
    ).T
    ig_df.index.name = 'drug'
    ig_path = os.path.join(args.outdir, 'drug_ig_attribution.csv')
    ig_df.to_csv(ig_path)
    print(f'\n[Output] IG attribution matrix → {ig_path}')

    topk_rows = []
    for drug, ig in ig_results.items():
        top_idx = np.argsort(np.abs(ig))[::-1][:args.topk]
        for rank, idx in enumerate(top_idx, 1):
            topk_rows.append({
                'drug': drug, 'rank': rank,
                'gene': gene_names[idx],
                'ig_score': float(ig[idx]),
                'direction': 'resistance' if ig[idx] > 0 else 'sensitivity',
            })
    topk_df = pd.DataFrame(topk_rows)
    topk_path = os.path.join(args.outdir, 'drug_top_genes.csv')
    topk_df.to_csv(topk_path, index=False)
    print(f'[Output] Top-{args.topk} genes per drug → {topk_path}')

    # ── Plots ──────────────────────────────────────────────────────────────
    print('\n[Plot] Building clustered attribution heatmap...')
    plot_attribution_heatmap(ig_results, gene_names, args.outdir, topk_per_drug=args.topk)

    print('[Plot] Building ranked bar charts...')
    plot_ranked_barplots(ig_results, gene_names, args.outdir, topk=args.topk)

    print('[Plot] Building GO pathway heatmap...')
    plot_pathway_heatmap(term_results, args.outdir, topk_terms=6)

    print('[Plot] Building drug PCA biplot...')
    plot_drug_pca(ig_results, gene_names, args.outdir)

    if args.with_string:
        print('\n[Network] Querying STRING and building network plots...')
        ncols  = 4
        nrows  = (len(ig_results) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6.5 * ncols, 6.5 * nrows),
                                 facecolor='white')
        axes = np.array(axes).flatten()

        for ax_idx, (drug, ig) in enumerate(ig_results.items()):
            ax = axes[ax_idx]
            top_idx     = np.argsort(np.abs(ig))[::-1][:args.topk]
            top_genes   = [gene_names[i].upper() for i in top_idx]
            gene_scores = {gene_names[i].upper(): float(ig[i]) for i in top_idx}
            string_edges = query_string_interactions(top_genes, args.score)
            G, linker_genes = build_drug_network(
                top_genes, gene_scores, string_edges, args.n_linkers
            )
            plot_drug_network(ax, drug, G, top_genes, linker_genes, gene_scores)

        for ax_idx in range(len(ig_results), len(axes)):
            axes[ax_idx].set_visible(False)

        legend_patches = [
            mpatches.Patch(color=BLUE, label='Sensitivity gene (negative score)'),
            mpatches.Patch(color=RED,  label='Resistance gene (positive score)'),
            mpatches.Patch(color=GRAY, label='Linker gene (STRING)'),
        ]
        fig.legend(handles=legend_patches, loc='lower center', ncol=3,
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.005))
        fig.suptitle(
            'Top-ranked genes as predictors for drug response\n'
            '(VNN GradCAM · OncoVNN · STRING network)',
            fontsize=11, fontweight='bold', y=0.995,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.985])
        plot_path = os.path.join(args.outdir, 'drug_gene_networks.png')
        plt.savefig(plot_path, dpi=180, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'[Output] Network plot           → {plot_path}')
    else:
        print('[Network] Skipped (pass --with_string to enable STRING network plot)')

    print('\nDone.')


if __name__ == '__main__':
    main()
