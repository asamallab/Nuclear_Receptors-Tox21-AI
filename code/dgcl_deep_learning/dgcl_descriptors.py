"""
DGCL: Dual GIN-GAT Contrastive Learning Architecture
Graph Neural Networks + Descriptors for Prediction
Validation-Based Threshold Selection with robust evaluation
"""

import sys
import copy
import itertools
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    precision_recall_curve, average_precision_score 
)
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GATv2Conv, global_mean_pool


# HPC CONTROL
num_cores = int(os.environ.get('OMP_NUM_THREADS', os.cpu_count()))
torch.set_num_threads(num_cores)
torch.set_num_interop_threads(num_cores)
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ARGUMENT CHECK
if len(sys.argv) != 7:
    print(
        "Usage: python3 dgcl_hpc_optimized.py "
        "<train_csv> <val_csv> <test_csv> <desc_csv> <checkpoint_dir> <output_txt>"
    )
    sys.exit(1)
train_path     = sys.argv[1]
val_path       = sys.argv[2]
test_path      = sys.argv[3]
desc_path      = sys.argv[4]
checkpoint_dir = sys.argv[5]
output_txt     = sys.argv[6]

os.makedirs(checkpoint_dir, exist_ok=True)
CHECKPOINT_CSV = os.path.join(checkpoint_dir, 'gin_gat_checkpoint.csv')
CHECKPOINT_PT  = os.path.join(checkpoint_dir, 'gin_gat_checkpoint.pt')
BEST_MODEL_PT  = os.path.join(checkpoint_dir, 'best_dgcl_gin_gat_model.pt')

# NODE AND EDGE FEATURE
def one_hot(value, choices):
    encoding = [0] * (len(choices) + 1)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding

ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br',
    'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
    'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd',
    'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',
    'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt',
    'Hg', 'Pb'
]

ATOM_DEGREES      = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
IMPLICIT_VALENCES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TOTAL_HS          = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
HYBRIDIZATIONS    = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

def atom_features(atom):
    return (
        one_hot(atom.GetSymbol(), ATOM_TYPES) +
        one_hot(atom.GetDegree(), ATOM_DEGREES) +
        one_hot(atom.GetImplicitValence(), IMPLICIT_VALENCES) +
        [atom.GetFormalCharge()] +
        [atom.GetNumRadicalElectrons()] +
        one_hot(atom.GetHybridization(), HYBRIDIZATIONS) +
        [int(atom.GetIsAromatic())] +
        one_hot(atom.GetTotalNumHs(), TOTAL_HS)
    )

NODE_FEATURE_DIM = (
    len(ATOM_TYPES) + 1 + len(ATOM_DEGREES) + 1 + len(IMPLICIT_VALENCES) + 1 +  1 +  1 + len(HYBRIDIZATIONS) + 1 +  1 +                            
    len(TOTAL_HS) + 1               
)
STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,   
    Chem.rdchem.BondStereo.STEREOANY,    
    Chem.rdchem.BondStereo.STEREOZ,     
    Chem.rdchem.BondStereo.STEREOE,     
    Chem.rdchem.BondStereo.STEREOCIS,    
    Chem.rdchem.BondStereo.STEREOTRANS, 
]  

def bond_features(bond):
    bond_type = bond.GetBondType()
    return (
        [
            int(bond_type == Chem.rdchem.BondType.SINGLE),
            int(bond_type == Chem.rdchem.BondType.DOUBLE),
            int(bond_type == Chem.rdchem.BondType.TRIPLE),
            int(bond_type == Chem.rdchem.BondType.AROMATIC),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
        ] + one_hot(bond.GetStereo(), STEREO_TYPES)   
    )
 
EDGE_FEATURE_DIM = 4 + 1 + 1 + (len(STEREO_TYPES) + 1) 

def smiles_to_graph(smiles, label=None, desc=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    node_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_feats, dtype=torch.float)
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        bf = bond_features(bond)
        edge_attr += [bf, bf]
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float)

    y = torch.tensor([label], dtype=torch.float) if label is not None else None
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    if desc is not None:
        data.desc = torch.tensor(np.array(desc, dtype=np.float32)).unsqueeze(0)
    return data

# LOAD DATA

train_df = pd.read_csv(train_path)
val_df   = pd.read_csv(val_path)
test_df  = pd.read_csv(test_path)

desc_raw = pd.read_csv(desc_path)
desc_id_col = desc_raw.columns[0]
desc_raw = desc_raw.rename(columns={desc_id_col: 'dsstox_substance_id'})

LABEL_COL = "activity_status"
for df in [train_df, val_df, test_df]:
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors='coerce')

# MERGE DATASETS 
def merge_and_extract(df, desc_df):
    merged = pd.merge(df[['dsstox_substance_id', LABEL_COL, 'SMILES']], desc_df, on='dsstox_substance_id', how='inner')
    drop_cols = ['dsstox_substance_id', LABEL_COL, 'SMILES']
    features = merged.drop(columns=drop_cols, errors='ignore').apply(pd.to_numeric, errors='coerce')
    return merged['SMILES'].tolist(), merged[LABEL_COL].astype(int).tolist(), features, merged['dsstox_substance_id'].tolist()

train_smiles, train_labels, train_desc, train_ids = merge_and_extract(train_df, desc_raw)
val_smiles,   val_labels,   val_desc, val_ids   = merge_and_extract(val_df,   desc_raw)
test_smiles,  test_labels,  test_desc, test_ids  = merge_and_extract(test_df,  desc_raw)

# DESCRIPTOR FILTERING 
nan_frac = train_desc.isna().mean(axis=0)
valid_cols = nan_frac[nan_frac <= 0.20].index.tolist()

train_desc = train_desc[valid_cols].fillna(train_desc[valid_cols].median())
val_desc   = val_desc[valid_cols].fillna(train_desc[valid_cols].median())
test_desc  = test_desc[valid_cols].fillna(train_desc[valid_cols].median())

selector = VarianceThreshold(threshold=0.01)
selector.fit(train_desc)
train_desc = train_desc.loc[:, selector.get_support()]
val_desc   = val_desc.loc[:, selector.get_support()]
test_desc  = test_desc.loc[:, selector.get_support()]

corr_matrix = train_desc.corr().abs()
upper_tri   = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop     = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]

train_desc = train_desc.drop(columns=to_drop)
val_desc   = val_desc.drop(columns=to_drop)
test_desc  = test_desc.drop(columns=to_drop)

FINAL_DESCRIPTOR_NAMES = train_desc.columns.tolist()
NUM_DESCRIPTORS        = len(FINAL_DESCRIPTOR_NAMES)

# SCALING
scaler = StandardScaler()
train_desc_scaled = scaler.fit_transform(train_desc.values.astype(np.float32))
val_desc_scaled   = scaler.transform(val_desc.values.astype(np.float32))
test_desc_scaled  = scaler.transform(test_desc.values.astype(np.float32))

# BUILD GRAPH DATASETS
def build_dataset(smiles_list, labels, desc_array, id_list):
    dataset, valid_smiles, valid_ids = [], [], []
    for i in range(len(smiles_list)):
        g = smiles_to_graph(smiles_list[i], labels[i], desc_array[i])
        if g is not None:
            dataset.append(g)
            valid_smiles.append(smiles_list[i])
            valid_ids.append(id_list[i]) 
    return dataset, valid_smiles, valid_ids

train_data, train_smiles_valid, train_ids_valid = build_dataset(train_smiles, train_labels, train_desc_scaled, train_ids)
val_data,   val_smiles_valid,   val_ids_valid   = build_dataset(val_smiles,   val_labels,   val_desc_scaled, val_ids)
test_data,  test_smiles_valid,  test_ids_valid  = build_dataset(test_smiles,  test_labels,  test_desc_scaled, test_ids)


# GNN ARCHITECTURE

# GIN ARCHITECTURE
class GINEncoder(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        def build_mlp(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        self.conv1 = GINEConv(build_mlp(input_dim),  edge_dim=edge_dim)
        self.conv2 = GINEConv(build_mlp(hidden_dim), edge_dim=edge_dim)
        self.conv3 = GINEConv(build_mlp(hidden_dim), edge_dim=edge_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.bn3   = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)

# GAT ARCHITECTURE
class GATEncoder(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim=128, dropout=0.3, heads=4):
        super().__init__()
        self.dropout = dropout
        head_dim     = hidden_dim // heads
        self.conv1 = GATv2Conv(input_dim,  head_dim, heads=heads, edge_dim=edge_dim, dropout=dropout, concat=True)
        self.conv2 = GATv2Conv(hidden_dim, head_dim, heads=heads, edge_dim=edge_dim, dropout=dropout, concat=True)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=1, edge_dim=edge_dim, dropout=dropout, concat=False)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.bn3   = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)

# MLP ARCHITECTURE
class DescriptorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, hidden_dim), nn.ReLU()
        )
    def forward(self, x): return self.mlp(x)

# PROJECTION HEAD
class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim))
    def forward(self, x): return self.mlp(x)

# DGCL DUAL ENCODER 
class DGCLModel(nn.Module):
    def __init__(self, input_dim, edge_dim, dropout):
        super().__init__()
        self.gin_encoder   = GINEncoder(input_dim, edge_dim, dropout=dropout)
        self.gat_encoder   = GATEncoder(input_dim, edge_dim, dropout=dropout)
        self.gin_projector = ProjectionHead()
        self.gat_projector = ProjectionHead()

    def forward(self, data1, data2):
      
        gin_h1 = self.gin_encoder(data1.x, data1.edge_index, data1.edge_attr, data1.batch)
        gin_h2 = self.gin_encoder(data2.x, data2.edge_index, data2.edge_attr, data2.batch)
        gin_z1, gin_z2 = self.gin_projector(gin_h1), self.gin_projector(gin_h2)

        gat_h1 = self.gat_encoder(data1.x, data1.edge_index, data1.edge_attr, data1.batch)
        gat_h2 = self.gat_encoder(data2.x, data2.edge_index, data2.edge_attr, data2.batch)
        gat_z1, gat_z2 = self.gat_projector(gat_h1), self.gat_projector(gat_h2)

        return gin_z1, gin_z2, gat_z1, gat_z2

class ClassifierWithDesc(nn.Module):
    def __init__(self, gin_encoder, gat_encoder, num_descriptors, dropout, graph_dim=128, desc_out_dim=64):
        super().__init__()
        self.gin_encoder = gin_encoder
        self.gat_encoder = gat_encoder
        self.desc_mlp    = DescriptorMLP(input_dim=num_descriptors, hidden_dim=desc_out_dim, dropout=dropout)
        fused_dim = graph_dim + graph_dim + desc_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.ReLU(),             nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.ReLU(),             nn.Dropout(dropout / 2),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        gin_emb    = self.gin_encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        gat_emb    = self.gat_encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        desc_input = data.desc.view(data.desc.size(0), -1)
        desc_emb   = self.desc_mlp(desc_input)
        fused      = torch.cat([gin_emb, gat_emb, desc_emb], dim=1)
        return self.classifier(fused)

# NT-XENT CONTRASTIVE LOSS
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        B = z1.size(0)
        z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        rep  = torch.cat([z1, z2], dim=0)
        sim  = torch.matmul(rep, rep.T) / self.temperature
        pos  = torch.zeros(2*B, 2*B, dtype=torch.bool, device=z1.device)
        pos[:B, B:] = torch.eye(B, device=z1.device).bool()
        pos[B:, :B] = torch.eye(B, device=z1.device).bool()
        sim.masked_fill_(torch.eye(2*B, dtype=torch.bool, device=z1.device), float('-inf'))
        return (-sim[pos] + torch.logsumexp(sim, dim=1)).mean()

# AUGMENTATIONS
def feature_mask(data, p=0.2):
    data = copy.deepcopy(data)
    data.x[torch.rand_like(data.x) < p] = 0
    return data

def edge_dropout(data, p=0.2):
    data = copy.deepcopy(data)
    if data.edge_index.size(1) > 0:
        mask = torch.rand(data.edge_index.size(1)) > p
        data.edge_index = data.edge_index[:, mask]
        data.edge_attr  = data.edge_attr[mask]
    return data

# GRID SEARCH
PARAM_GRID = {
    'batch_size'   : [32, 64],
    'lr'           : [1e-3, 5e-4, 1e-4],
    'dropout'      : [0.2, 0.3],
    'weight_decay' : [0.0, 1e-4],
}

PRETRAIN_EPOCHS = 50
FINETUNE_EPOCHS = 30
POS_WEIGHT      = 2.5
TEMPERATURE     = 0.5

param_keys = list(PARAM_GRID.keys())
all_combos = list(itertools.product(*PARAM_GRID.values()))

loaders = {}
for bs in PARAM_GRID['batch_size']:
    loaders[bs] = {
        'train': DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=0),
        'val':   DataLoader(val_data,   batch_size=bs, num_workers=0),
    }

#EVALUATION
def evaluate_model(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data  = data.to(device)
            preds = torch.sigmoid(model(data).squeeze(-1))
            if preds.dim() == 0: preds = preds.unsqueeze(0)
            all_probs.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    pr_precision, pr_recall, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (pr_precision[:-1] * pr_recall[:-1]) / (pr_precision[:-1] + pr_recall[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_t = thresholds[best_idx] if len(thresholds) > 0 else 0.5
    
    preds   = (all_probs >= best_t).astype(int)
    acc     = accuracy_score(all_labels, preds)
    prec    = precision_score(all_labels, preds, zero_division=0)
    rec     = recall_score(all_labels, preds, zero_division=0)
    f1      = f1_score(all_labels, preds, zero_division=0)
    mcc     = matthews_corrcoef(all_labels, preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc  = average_precision_score(all_labels, all_probs)
    
    return {'threshold': best_t, 'accuracy': acc, 'precision': prec, 'recall': rec, 
            'f1': f1, 'mcc': mcc, 'roc_auc': roc_auc, 'pr_auc': pr_auc,    
        'y_true': all_labels, 
        'y_prob': all_probs,  
        'y_pred': preds}

grid_results, start_trial = [], 0
best_val_f1, best_threshold = -1, 0.5
best_params, best_classifier_state, best_pretrain_state = None, None, None

if os.path.exists(CHECKPOINT_CSV) and os.path.exists(CHECKPOINT_PT):
    
    grid_results = pd.read_csv(CHECKPOINT_CSV).to_dict('records')
    start_trial  = len(grid_results)
    ckpt         = torch.load(CHECKPOINT_PT, map_location=device, weights_only=False)
    best_val_f1  = ckpt['best_val_f1']
    best_params  = ckpt['best_params']
    best_threshold = ckpt['best_threshold']
    best_classifier_state = ckpt['classifier_state_dict']
    best_pretrain_state   = ckpt['pretrain_state_dict']

pos_weight_tensor = torch.tensor([POS_WEIGHT]).to(device)
nt_xent = NTXentLoss(temperature=TEMPERATURE)

global_start = time.time()

for trial_num, combo in enumerate(all_combos):
    if trial_num < start_trial: continue

    params       = dict(zip(param_keys, combo))
    batch_size   = params['batch_size']
    lr           = params['lr']
    dropout      = params['dropout']
    weight_decay = params['weight_decay']

    t_start = time.time()
    t_loader = loaders[batch_size]['train']
    v_loader = loaders[batch_size]['val']

    pretrain_model = DGCLModel(input_dim=NODE_FEATURE_DIM, edge_dim=EDGE_FEATURE_DIM, dropout=dropout).to(device)
    pretrain_opt   = torch.optim.Adam(pretrain_model.parameters(), lr=lr, weight_decay=weight_decay)

    pretrain_model.train()
    for _ in range(PRETRAIN_EPOCHS):
        for data in t_loader:
            data = data.to(device)
            aug1, aug2 = feature_mask(data), edge_dropout(data)
            pretrain_opt.zero_grad()
            gin_z1, gin_z2, gat_z1, gat_z2 = pretrain_model(aug1, aug2)
            loss = nt_xent(gin_z1, gin_z2) + nt_xent(gat_z1, gat_z2)
            loss.backward()
            pretrain_opt.step()

    classifier = ClassifierWithDesc(
        gin_encoder=pretrain_model.gin_encoder, gat_encoder=pretrain_model.gat_encoder,
        num_descriptors=NUM_DESCRIPTORS, dropout=dropout
    ).to(device)
    
    finetune_opt = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    bce_loss     = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    classifier.train()
    for _ in range(FINETUNE_EPOCHS):
        for data in t_loader:
            data = data.to(device)
            finetune_opt.zero_grad()
            out = classifier(data).squeeze(-1)
            if out.dim() == 0: out = out.unsqueeze(0)
            bce_loss(out, data.y.float()).backward()
            finetune_opt.step()

    val_metrics = evaluate_model(classifier, v_loader)
    val_f1      = val_metrics['f1']
    val_thresh  = val_metrics['threshold']
    elapsed     = time.time() - t_start
    

    grid_results.append({
        'trial': trial_num + 1, 'batch_size': batch_size, 'lr': lr, 'dropout': dropout,
        'weight_decay': weight_decay, 'val_f1': round(val_f1, 4),
        'val_threshold': round(val_thresh, 2), 'time_sec': round(elapsed, 1),
    })
    pd.DataFrame(grid_results).to_csv(CHECKPOINT_CSV, index=False)

    if val_f1 > best_val_f1:
        best_val_f1, best_params, best_threshold = val_f1, params.copy(), val_thresh
        best_classifier_state = copy.deepcopy(classifier.state_dict())
        best_pretrain_state   = copy.deepcopy(pretrain_model.state_dict())

        torch.save({
            'classifier_state_dict': best_classifier_state, 'pretrain_state_dict': best_pretrain_state,
            'best_params': best_params, 'best_threshold': best_threshold,
            'best_val_f1': best_val_f1, 'completed_trials': trial_num + 1,
        }, CHECKPOINT_PT)



# TEST EVALUATION

global_end = time.time() - global_start 

best_pretrain_model = DGCLModel(
    input_dim=NODE_FEATURE_DIM,
    edge_dim=EDGE_FEATURE_DIM,
    dropout=best_params['dropout']
).to(device)
best_pretrain_model.load_state_dict(best_pretrain_state)

best_classifier = ClassifierWithDesc(
    gin_encoder=best_pretrain_model.gin_encoder,
    gat_encoder=best_pretrain_model.gat_encoder,
    num_descriptors=NUM_DESCRIPTORS,
    dropout=best_params['dropout']
).to(device)

best_classifier.load_state_dict(best_classifier_state)

test_loader = DataLoader(
    test_data,
    batch_size=best_params['batch_size'],
    num_workers=0
)

best_classifier.eval()
all_probs, all_labels = [], []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        logits = best_classifier(data).squeeze(-1)
        probs = torch.sigmoid(logits)

        if probs.dim() == 0:
            probs = probs.unsqueeze(0)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)

test_preds = (all_probs >= best_threshold).astype(int)

test_metrics = {
    'threshold': best_threshold,
    'accuracy' : accuracy_score(all_labels, test_preds),
    'precision': precision_score(all_labels, test_preds, zero_division=0),
    'recall'   : recall_score(all_labels, test_preds, zero_division=0),
    'f1'       : f1_score(all_labels, test_preds, zero_division=0),
    'mcc'      : matthews_corrcoef(all_labels, test_preds),
    'roc_auc'  : roc_auc_score(all_labels, all_probs),
    'pr_auc'   : average_precision_score(all_labels, all_probs),
    'y_true'   : all_labels,
    'y_prob'   : all_probs,
    'y_pred'   : test_preds
}


# SAVE RESULTS

torch.save({
    'classifier_state_dict': best_classifier_state,
    'pretrain_state_dict': best_pretrain_state,
    'best_params': best_params,
    'best_threshold': best_threshold,
    'num_descriptors': NUM_DESCRIPTORS,
    'descriptor_names': FINAL_DESCRIPTOR_NAMES,
    'test_metrics': test_metrics,
}, BEST_MODEL_PT)

with open(output_txt, "w") as f:
    f.write("DGCL GIN+GAT EDC Classification Results\n")
    f.write("="*60 + "\n")
    f.write(f"Training samples: {len(train_data)}\n")
    f.write(f"Validation samples: {len(val_data)}\n")
    f.write(f"Test samples: {len(test_data)}\n")
    f.write(f"Descriptors used: {NUM_DESCRIPTORS}\n")
    f.write("\n")
    f.write("Test Set Performance:\n")
    f.write("-" * 60 + "\n")
    f.write(f"Best Threshold : {test_metrics['threshold']:.4f}\n")
    f.write(f"Accuracy       : {test_metrics['accuracy']:.4f}\n")
    f.write(f"Precision      : {test_metrics['precision']:.4f}\n")
    f.write(f"Recall         : {test_metrics['recall']:.4f}\n")
    f.write(f"F1-score       : {test_metrics['f1']:.4f}\n")
    f.write(f"MCC            : {test_metrics['mcc']:.4f}\n")
    f.write(f"ROC-AUC        : {test_metrics['roc_auc']:.4f}\n")
    f.write(f"PR-AUC         : {test_metrics['pr_auc']:.4f}\n")
    f.write("\n")
    f.write(f"Best Hyperparameters: {best_params}\n")
    f.write(f"Total processing time: {global_end:.2f} seconds ({global_end/60:.2f} minutes)\n")
    f.write("="*60 + "\n")
