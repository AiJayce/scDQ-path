from pathlib import Path
from copy import deepcopy
import ast
import json
import random
import shutil
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch import nn
from sklearn.decomposition import PCA
from IPython.display import display


ROOT = Path(__file__).resolve().parent
notebook = json.loads((ROOT / 'scDQN.ipynb').read_text())
sources = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
result_dir = ROOT / 'scDQN_results/coordinate_dqn'
if not list(result_dir.parent.glob('coordinate_dqn_before_reward_v2_*')):
    backup_dir = result_dir.parent / f'coordinate_dqn_before_reward_v2_{time.strftime("%Y%m%d_%H%M%S")}'
    shutil.copytree(result_dir, backup_dir)
    print(f'Previous results backed up: {backup_dir}', flush=True)
SEED, LATENT_DIM = 42, 16
SUBTYPES = ['CD4 Naive T', 'Regulatory T']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(result_dir / 'coordinate_models.pt', map_location='cpu', weights_only=False)
with np.load(result_dir / 'coordinate_graph.npz', allow_pickle=False) as saved:
    Z_space = saved['Z_space'].copy()
    barcodes = saved['cell_barcodes'].astype(str)
model_source = next(source for source in sources if 'class ExpressionVAE' in source)
model_class = next(node for node in ast.parse(model_source).body if isinstance(node, ast.ClassDef))
exec(compile(ast.Module(body=[model_class], type_ignores=[]), '<saved-vae-class>', 'exec'))
vae = ExpressionVAE(checkpoint['vae']['encoder.0.weight'].shape[1], LATENT_DIM).to(DEVICE)
vae.load_state_dict(checkpoint['vae'])
vae.eval().requires_grad_(False)
pca = PCA(n_components=2)
pca.components_ = checkpoint['pca_components']
pca.mean_ = checkpoint['pca_mean']
pca.n_features_in_ = LATENT_DIM
pca.n_components_ = 2
pca.explained_variance_ = np.var((Z_space - pca.mean_) @ pca.components_.T, axis=0, ddof=1)
pca.explained_variance_ratio_ = np.var((Z_space - pca.mean_) @ pca.components_.T, axis=0, ddof=1) / np.var(Z_space, axis=0, ddof=1).sum()
Z_pca = pca.transform(Z_space)
reference = sc.read_h5ad('/home/Data_Drive_8TB/kykim/8. RL/tdata_sub.h5ad', backed='r')
try:
    adata = sc.AnnData(obs=reference.obs.loc[barcodes].copy())
finally:
    reference.file.close()
for anchor in (
    'def transition_reward_components',
    'class LatentQNetwork',
    'rl_start_time = time.time()',
    'def entropy_greedy_rollout',
    'virtual_steps = rl_path[1:-1]',
    'def update_pca_video',
):
    source = next(source for source in sources if anchor in source)
    print(f'Running notebook section: {anchor}', flush=True)
    exec(compile(source, str(ROOT / 'scDQN.ipynb') + ':' + anchor, 'exec'))
    plt.close('all')
for key, value in checkpoint['vae'].items():
    torch.testing.assert_close(vae.state_dict()[key].cpu(), value, rtol=0, atol=0)
np.testing.assert_array_equal(pca.components_, checkpoint['pca_components'])
np.testing.assert_array_equal(pca.mean_, checkpoint['pca_mean'])
print('COMPLETE: DQN retrained, coordinates/rewards exported, video decoded; original VAE/PCA unchanged.', flush=True)