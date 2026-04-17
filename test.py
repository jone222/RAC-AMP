import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    precision_recall_curve, auc
import numpy as np
import os
import pickle
import logging
from datetime import datetime
from torch.nn import Linear, Dropout, Sequential, ReLU, LayerNorm
import tqdm
from typing import Tuple
from torch_geometric.utils import to_dense_batch
import random


from torch.amp import autocast, GradScaler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
MODEL_SAVE_PATH = "32b"  # 5个模型保存在这个文件夹下
THRESHOLD = 0.5
RES_DIM = 1280
ATOM_DIM = 21
HIDDEN_DIM = 128
NUM_HEADS = 4
DROPOUT_RATE = 0.2


# --------------------------- 1. 全局日志配置 ---------------------------
def setup_logging():
    os.makedirs("test_logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"test_logs/test_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# --------------------------- 残基自注意力模块 ---------------------------
class ResidueSelfAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=4, dropout_rate=0.2):
        super().__init__()
        self.proj = Linear(in_dim, hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        self.ffn = Sequential(
            Linear(hidden_dim, hidden_dim * 2),
            ReLU(),
            Dropout(dropout_rate),
            Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x, batch):
        dense_x, mask = to_dense_batch(x, batch)
        dense_x = self.proj(dense_x)
        attn_out, _ = self.self_attn(dense_x, dense_x, dense_x, key_padding_mask=~mask)
        dense_x = self.norm1(dense_x + self.dropout(attn_out))
        ffn_out = self.ffn(dense_x)
        dense_x = self.norm2(dense_x + self.dropout(ffn_out))
        return dense_x[mask]


# --------------------------- EGNN层 ---------------------------
class EGNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.msg_mlp = Sequential(
            Linear(in_dim * 2 + 1, out_dim * 2),
            ReLU(),
            Dropout(dropout),
            Linear(out_dim * 2, out_dim)
        )
        self.update_mlp = Sequential(
            Linear(out_dim * 2, out_dim * 2),
            ReLU(),
            Dropout(dropout),
            Linear(out_dim * 2, out_dim)
        )
        self.projection = Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.msg_mlp:
            if isinstance(m, Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.update_mlp:
            if isinstance(m, Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        if isinstance(self.projection, Linear):
            nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        N = x.shape[0]
        edge_i, edge_j = edge_index[0], edge_index[1]
        if torch.any(edge_i >= N) or torch.any(edge_j >= N):
            valid_mask = (edge_i < N) & (edge_j < N)
            edge_i = edge_i[valid_mask]
            edge_j = edge_j[valid_mask]
            if len(edge_i) == 0:
                edge_i = torch.tensor([0], device=x.device)
                edge_j = torch.tensor([0], device=x.device)
        pos_ij = pos[edge_j] - pos[edge_i]
        dist_ij = torch.norm(pos_ij, p=2, dim=1, keepdim=True)
        x_i, x_j = x[edge_i], x[edge_j]
        msg_input = torch.cat([x_i, x_j, dist_ij], dim=1)
        msg = self.msg_mlp(msg_input)
        agg_msg = torch.zeros(N, self.out_dim, device=x.device, dtype=msg.dtype)
        agg_msg.index_add_(0, edge_i, msg)
        x_proj = self.projection(x)
        update_input = torch.cat([x_proj, agg_msg], dim=1)
        x_out = x_proj + self.update_mlp(update_input)
        x_out = F.elu(x_out)
        pos_weight = torch.tanh(torch.norm(msg, dim=1, keepdim=True))
        edge_pos_update = pos_weight * pos_ij
        pos_update_sum = torch.zeros_like(pos)
        pos_update_sum.index_add_(0, edge_i, edge_pos_update)
        neighbor_counts = torch.zeros(N, device=x.device)
        neighbor_counts.index_add_(0, edge_i, torch.ones_like(edge_i, dtype=torch.float))
        neighbor_counts = neighbor_counts.clamp(min=1.0).unsqueeze(1)
        pos_out = pos + pos_update_sum / neighbor_counts
        return x_out, pos_out


# --------------------------- 双通道分类器模型 ---------------------------
class DualChannelAMPClassifier(nn.Module):
    def __init__(self, res_dim=1280, atom_dim=21, hidden_dim=128, num_heads=4, dropout_rate=0.2):
        super(DualChannelAMPClassifier, self).__init__()
        self.res_proj = nn.Sequential(
            nn.Linear(res_dim, hidden_dim),
            ReLU(),
            Dropout(dropout_rate)
        )
        self.res_attn = ResidueSelfAttention(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        self.atom_egnn1 = EGNNLayer(atom_dim, hidden_dim, dropout=dropout_rate)
        self.atom_egnn2 = EGNNLayer(hidden_dim, hidden_dim, dropout=dropout_rate)
        self.atom_egnn3 = EGNNLayer(hidden_dim, hidden_dim, dropout=dropout_rate)
        self.atom_proj = Linear(hidden_dim, hidden_dim)
        self.res_global_proj = Linear(2 * hidden_dim, hidden_dim)
        self.atom_global_proj = Linear(2 * hidden_dim, hidden_dim)
        self.qkv_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        self.fusion_adjust = Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            ReLU(),
            Dropout(dropout_rate)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            nn.Dropout(dropout_rate),
            Linear(hidden_dim // 2, 1)
        )

    def forward(self, res_data, atom_data):
        res_x, res_batch = res_data.x, res_data.batch
        res_x = self.res_proj(res_x)
        res_x = self.res_attn(res_x, res_batch)
        res_mean = global_mean_pool(res_x, res_batch)
        res_max = global_max_pool(res_x, res_batch)
        res_global = torch.cat([res_mean, res_max], dim=1)
        res_global = self.res_global_proj(res_global)

        atom_x, atom_pos, atom_edge_index, atom_batch = (
            atom_data.x, atom_data.pos, atom_data.edge_index, atom_data.batch
        )
        atom_x, atom_pos = self.atom_egnn1(atom_x, atom_pos, atom_edge_index)
        atom_x, atom_pos = self.atom_egnn2(atom_x, atom_pos, atom_edge_index)
        atom_x, atom_pos = self.atom_egnn3(atom_x, atom_pos, atom_edge_index)
        atom_x = self.atom_proj(atom_x)
        atom_x = F.elu(atom_x)
        atom_mean = global_mean_pool(atom_x, atom_batch)
        atom_max = global_max_pool(atom_x, atom_batch)
        atom_global = torch.cat([atom_mean, atom_max], dim=1)
        atom_global = self.atom_global_proj(atom_global)

        Q = res_global.unsqueeze(1)
        K = atom_global.unsqueeze(1)
        V = atom_global.unsqueeze(1)
        attn_output, attn_weights = self.qkv_attention(Q, K, V)
        attn_output = attn_output.squeeze(1)
        fused = torch.cat([res_global, attn_output], dim=1)
        fused = self.fusion_adjust(fused)
        amp_logits = self.classifier(fused)
        return amp_logits


# --------------------------- 双通道数据集类 ---------------------------
class DualChannelDataset(Dataset):
    def __init__(self, residue_feats, atom_feats, labels, expected_res_dim=1280, expected_atom_dim=21):
        self.residue_feats = residue_feats
        self.atom_feats = atom_feats
        self.labels = labels
        self.expected_res_dim = expected_res_dim
        self.expected_atom_dim = expected_atom_dim

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        res_raw = self.residue_feats[idx]
        atom_raw = self.atom_feats[idx]
        label = torch.tensor([self.labels[idx]], dtype=torch.float)

        if isinstance(res_raw, list):
            res_list = []
            for res in res_raw:
                flattened = [float(x) for x in res]
                if len(flattened) > self.expected_res_dim:
                    res_list.append(flattened[:self.expected_res_dim])
                else:
                    res_list.append(flattened + [0.0] * (self.expected_res_dim - len(flattened)))
            res_feat = np.array(res_list, dtype=np.float32)
        elif isinstance(res_raw, np.ndarray):
            if res_raw.ndim == 1:
                res_feat = res_raw.reshape(1, -1)
            else:
                res_feat = res_raw
            if res_feat.shape[1] != self.expected_res_dim:
                if res_feat.shape[1] > self.expected_res_dim:
                    res_feat = res_feat[:, :self.expected_res_dim]
                else:
                    pad_width = ((0, 0), (0, self.expected_res_dim - res_feat.shape[1]))
                    res_feat = np.pad(res_feat, pad_width, mode="constant", constant_values=0.0)
        else:
            res_feat = np.zeros((1, self.expected_res_dim), dtype=np.float32)

        res_data = Data(x=torch.tensor(res_feat, dtype=torch.float), y=label)

        atom_features = []
        atom_positions = []
        if isinstance(atom_raw, dict):
            if 'node_features' in atom_raw and 'positions' in atom_raw:
                atom_features = atom_raw['node_features']
                atom_positions = atom_raw['positions']
        elif isinstance(atom_raw, list):
            for aa_data in atom_raw:
                if isinstance(aa_data, dict):
                    if 'node_features' in aa_data and 'positions' in aa_data:
                        atom_features.extend(aa_data['node_features'])
                        atom_positions.extend(aa_data['positions'])
                elif isinstance(aa_data, (list, np.ndarray)):
                    for atom in aa_data:
                        if len(atom) >= self.expected_atom_dim + 3:
                            atom_features.append(atom[:self.expected_atom_dim])
                            atom_positions.append(atom[self.expected_atom_dim:self.expected_atom_dim + 3])
        if len(atom_features) == 0:
            atom_features = [np.zeros(self.expected_atom_dim, dtype=np.float32)]
            atom_positions = [np.zeros(3, dtype=np.float32)]

        atom_feat_array = np.array(atom_features, dtype=np.float32)
        atom_pos_array = np.array(atom_positions, dtype=np.float32)
        if atom_feat_array.shape[1] != self.expected_atom_dim:
            if atom_feat_array.shape[1] > self.expected_atom_dim:
                atom_feat_array = atom_feat_array[:, :self.expected_atom_dim]
            else:
                pad_width = ((0, 0), (0, self.expected_atom_dim - atom_feat_array.shape[1]))
                atom_feat_array = np.pad(atom_feat_array, pad_width, mode="constant", constant_values=0.0)

        num_atoms = atom_feat_array.shape[0]
        if num_atoms > 1:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min(8, num_atoms)).fit(atom_pos_array)
            distances, indices = nbrs.kneighbors(atom_pos_array)
            rows = []
            cols = []
            for i, neighbors in enumerate(indices):
                for j in neighbors:
                    if i != j:
                        rows.append(i)
                        cols.append(j)
            atom_edge_index = np.array([rows, cols], dtype=np.int64)
        else:
            atom_edge_index = np.array([[0], [0]], dtype=np.int64)

        valid_mask = (atom_edge_index[0] < num_atoms) & (atom_edge_index[1] < num_atoms)
        atom_edge_index = atom_edge_index[:, valid_mask]
        if atom_edge_index.shape[1] == 0 and num_atoms > 0:
            atom_edge_index = np.array([[0], [0]], dtype=np.int64)

        atom_data = Data(
            x=torch.tensor(atom_feat_array, dtype=torch.float),
            pos=torch.tensor(atom_pos_array, dtype=torch.float),
            edge_index=torch.tensor(atom_edge_index, dtype=torch.long),
            y=label
        )
        return (res_data, atom_data, label)


# --------------------------- collate_fn ---------------------------
def dual_collate_fn(batch):
    try:
        res_data_list = []
        atom_data_list = []
        labels = []
        for res_data, atom_data, label in batch:
            res_data_list.append(res_data)
            atom_data_list.append(atom_data)
            labels.append(label)
        res_batch = Batch.from_data_list(res_data_list)
        atom_batch = Batch.from_data_list(atom_data_list)
        labels = torch.cat(labels, dim=0)
        return (res_batch, atom_batch, labels)
    except Exception as e:
        logger.error(f"Collate Error: {e}")
        return (
            Batch.from_data_list([Data(x=torch.zeros(1, 1280), y=torch.zeros(1))]),
            Batch.from_data_list([Data(x=torch.zeros(1, 21), pos=torch.zeros(1, 3), edge_index=torch.zeros(2, 1).long(),
                                       y=torch.zeros(1))]),
            torch.tensor([[0.0]]))


# --------------------------- 加载测试集数据 ---------------------------
def load_feat(path, feat_type):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            return list(data.values())
        elif isinstance(data, list):
            return data
        return []
    except Exception as e:
        logger.warning(f"加载{feat_type}特征失败: {e}")
        return []



def calculate_metrics(true_labels, pred_probs, threshold=0.5):
    pred_labels = (pred_probs > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(true_labels, pred_probs)
        aucpr = auc(recall_curve, precision_curve)
    except:
        aucpr = 0.0
    metrics = {
        "acc": accuracy_score(true_labels, pred_labels),
        "prec": precision_score(true_labels, pred_labels, zero_division=0),
        "recall": recall_score(true_labels, pred_labels, zero_division=0),
        "f1": f1_score(true_labels, pred_labels, zero_division=0),
        "auc": roc_auc_score(true_labels, pred_probs) if len(set(true_labels)) > 1 else 0.0,
        "sp": specificity,
        "aucpr": aucpr,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }
    return metrics



def test_individual_models():
    # ====================== 【测试集路径】 ======================
    test_data_paths = {
        # 残基特征文件
        "res_pos_test": r"",
        "res_neg_test": r"",
        # 原子特征文件
        "atom_pos_test": r"",
        "atom_neg_test": r"",
    }
    # ==========================================================

    # 固定随机种子
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # 加载测试集数据
    res_pos_test = load_feat(test_data_paths["res_pos_test"], "测试集残基正例")
    res_neg_test = load_feat(test_data_paths["res_neg_test"], "测试集残基负例")
    atom_pos_test = load_feat(test_data_paths["atom_pos_test"], "测试集原子正例")
    atom_neg_test = load_feat(test_data_paths["atom_neg_test"], "测试集原子负例")

    # 合并测试集数据
    test_res_feats = list(res_pos_test) + list(res_neg_test)
    test_atom_feats = list(atom_pos_test) + list(atom_neg_test)
    test_labels = [1] * len(res_pos_test) + [0] * len(res_neg_test)
    total_test_samples = len(test_labels)

    # 构建测试集加载器
    test_dataset = DualChannelDataset(test_res_feats, test_atom_feats, test_labels, RES_DIM, ATOM_DIM)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=dual_collate_fn, num_workers=0)

    # 加载5个折的最优模型
    models = []
    for fold_idx in range(1, 6):
        model_path = os.path.join(MODEL_SAVE_PATH, f"shiyan_fold{fold_idx}_best.pth")
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return
        model = DualChannelAMPClassifier(res_dim=RES_DIM, atom_dim=ATOM_DIM, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS,
                                         dropout_rate=DROPOUT_RATE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE).eval()
        models.append(model)
    logger.info(f" 成功加载5个模型，开始独立测试！")

    # 准备存储各个模型的预测结果和损失
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    per_model_total_loss = [0.0 for _ in range(5)]
    per_model_all_probs = [[] for _ in range(5)]
    all_true_labels = []
    use_amp = DEVICE.type == 'cuda'

    with torch.no_grad():
        progress_bar = tqdm.tqdm(test_loader, desc="测试中", leave=False)
        for res_batch, atom_batch, labels in progress_bar:
            res_batch, atom_batch = res_batch.to(DEVICE), atom_batch.to(DEVICE)
            labels_cuda = labels.to(DEVICE).unsqueeze(1)
            all_true_labels.extend(labels.numpy().flatten())

            for model_idx, model in enumerate(models):
                with autocast('cuda', enabled=use_amp, dtype=torch.float16):
                    logits = model(res_batch, atom_batch)
                    prob = torch.sigmoid(logits).cpu().numpy().flatten()

                per_model_all_probs[model_idx].extend(prob)
                loss_val = criterion(logits, labels_cuda).item()
                per_model_total_loss[model_idx] += loss_val * labels.size(0)

    # 计算各折独立指标
    all_true_labels = np.array(all_true_labels)
    model_metrics_list = []
    for fold_idx in range(5):
        m = calculate_metrics(all_true_labels, np.array(per_model_all_probs[fold_idx]), THRESHOLD)
        m['loss'] = per_model_total_loss[fold_idx] / total_test_samples
        model_metrics_list.append(m)

    # 计算均值和标准差
    metrics_keys = ['loss', 'acc', 'prec', 'recall', 'f1', 'auc', 'sp', 'aucpr']
    count_keys = ['tp', 'tn', 'fp', 'fn']
    fold_avg = {key: np.mean([m[key] for m in model_metrics_list]) for key in metrics_keys + count_keys}
    fold_std = {key: np.std([m[key] for m in model_metrics_list], ddof=1) for key in metrics_keys + count_keys}

    # 打印汇总表
    logger.info("\n" + "=" * 150)
    logger.info("五折独立模型 - 测试集结果汇总表")
    logger.info("=" * 150)
    header = f"{'模型':<12}{'Loss':<18}{'Acc':<12}{'Prec':<12}{'Sn':<12}{'F1':<12}{'AUC':<12}{'SP':<12}{'AUCPR':<12}"
    logger.info(header)
    logger.info("-" * 150)
    for i, m in enumerate(model_metrics_list):
        logger.info(
            f"Fold-{i + 1:<11}{m['loss']:<18.6f}{m['acc']:<12.4f}{m['prec']:<12.4f}{m['recall']:<12.4f}{m['f1']:<12.4f}{m['auc']:<12.4f}{m['sp']:<12.4f}{m['aucpr']:<12.4f}")

    logger.info("-" * 150)
    res_str = lambda k, fmt: f"{fold_avg[k]:{fmt}} ± {fold_std[k]:{fmt}}"
    line = f"五折均值±标差 {res_str('loss', '.6f'):<18}{res_str('acc', '.4f'):<12}{res_str('prec', '.4f'):<12}{res_str('recall', '.4f'):<12}{res_str('f1', '.4f'):<12}{res_str('auc', '.4f'):<12}{res_str('sp', '.4f'):<12}{res_str('aucpr', '.4f'):<12}"
    logger.info(line)
    logger.info("=" * 150)

    logger.info(
        f"\n平均混淆矩阵 | TP: {res_str('tp', '.2f')} | TN: {res_str('tn', '.2f')} | FP: {res_str('fp', '.2f')} | FN: {res_str('fn', '.2f')}")
    logger.info("=" * 150)


if __name__ == "__main__":
    test_individual_models()