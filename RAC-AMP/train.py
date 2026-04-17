import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    precision_recall_curve, auc

from sklearn.model_selection import KFold
import numpy as np
import os
import pickle
import logging
from datetime import datetime
from torch.nn import Linear, Dropout, Sequential, ReLU, LayerNorm
import torch.optim as optim
# 删除了调度器导入 (不删除也不影响)
from tqdm import tqdm
from typing import Tuple
from torch_geometric.utils import to_dense_batch
import random

from torch.amp import autocast, GradScaler


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_{timestamp}.log"

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


class EGNNLayer(nn.Module):
    """标准EGNN层，包含等变消息传递和坐标更新"""

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


# --------------------------- 3. 双通道分类器模型定义 ---------------------------
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
            nn.Linear(hidden_dim // 2, 1)
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


# --------------------------- 4. 双通道数据集类 ---------------------------
class DualChannelDataset(Dataset):
    def __init__(self, residue_feats, atom_feats, labels, expected_res_dim=1280, expected_atom_dim=21):
        self.residue_feats = residue_feats
        self.atom_feats = atom_feats
        self.labels = labels
        self.expected_res_dim = expected_res_dim
        self.expected_atom_dim = expected_atom_dim
        logger.info(f"双通道数据集初始化 | 样本数：{len(labels)}")

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

        res_data = Data(
            x=torch.tensor(res_feat, dtype=torch.float),
            y=label
        )

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
        logger.error(f"Error: {e}")
        return (
            Batch.from_data_list([Data(x=torch.zeros(1, 1280), y=torch.zeros(1))]),
            Batch.from_data_list([Data(x=torch.zeros(1, 21), pos=torch.zeros(1, 3), edge_index=torch.zeros(2, 1).long(),
                                       y=torch.zeros(1))]),
            torch.tensor([[0.0]]))


# --------------------------- 6. 训练函数 ---------------------------
def train(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    total_samples = 0

    use_amp = scaler is not None and device.type == 'cuda'

    progress_bar = tqdm(train_loader, desc="训练中", leave=False)
    for batch_idx, (res_batch, atom_batch, labels) in enumerate(progress_bar):
        res_batch = res_batch.to(device)
        atom_batch = atom_batch.to(device)
        labels = labels.to(device).view(-1, 1)

        optimizer.zero_grad()

        with autocast('cuda', enabled=use_amp, dtype=torch.float16):
            outputs = model(res_batch, atom_batch)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        progress_bar.set_postfix({
            "损失": f"{loss.item():.6f}",
            "平均损失": f"{total_loss / total_samples:.6f}" if total_samples > 0 else "N/A"
        })

    if total_samples == 0:
        return 0.0
    return total_loss / total_samples


# --------------------------- 7. 评估函数 ---------------------------
def evaluate(model, val_loader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    val_loss = 0.0
    total_samples = 0

    use_amp = device.type == 'cuda'

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="验证中", leave=False)
        for res_batch, atom_batch, labels in progress_bar:
            res_batch = res_batch.to(device)
            atom_batch = atom_batch.to(device)
            labels = labels.to(device).view(-1, 1)

            with autocast('cuda', enabled=use_amp, dtype=torch.float16):
                outputs = model(res_batch, atom_batch)
                loss = F.binary_cross_entropy_with_logits(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            probs = torch.sigmoid(outputs).float().cpu().numpy()
            preds = (probs > threshold).astype(int)

            all_preds.extend(preds.flatten())
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

            progress_bar.set_postfix({"损失": f"{loss.item():.6f}"})

    if total_samples == 0:
        return {"acc": 0.0, "prec": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0, "sp": 0.0, "aucpr": 0.0, "loss": 0.0}

    avg_val_loss = val_loss / total_samples

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
        aucpr = auc(recall_curve, precision_curve)
    except:
        aucpr = 0.0

    metrics = {
        "acc": accuracy_score(all_labels, all_preds),
        "prec": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
        "sp": specificity,
        "aucpr": aucpr,
        "loss": avg_val_loss
    }

    logger.info(
        f"验证指标 | Acc: {metrics['acc']:.4f} | Prec: {metrics['prec']:.4f} | "
        f"Recall(Sn): {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | "
        f"AUC: {metrics['auc']:.4f} | SP: {metrics['sp']:.4f} | AUCPR: {metrics['aucpr']:.4f} | "
        f"Loss: {metrics['loss']:.6f}"
    )

    return metrics


# --------------------------- 8. 主训练流程 ---------------------------
def main():

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("=" * 60)
    logger.info("开始双通道AMP分类模型训练（五折交叉验证）")
    logger.info("=" * 60)

    config = {
        "lr": 1e-3,
        "epochs": 100,
        "batch_size": 32,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "expected_res_dim": 1280,
        "expected_atom_dim": 21,
        "hidden_dim": 128,
        "num_heads": 4,
        "dropout_rate": 0.2,
        "threshold": 0.5,
        "early_stopping_patience": 10

    }

    data_paths = {
        #残基级特征
        "res_pos_train": r"",
        "res_neg_train": r"",
        #原子级特征
        "atom_pos_train": r"",
        "atom_neg_train": r"",
    }

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

    res_pos = load_feat(data_paths["res_pos_train"], "残基正例")
    res_neg = load_feat(data_paths["res_neg_train"], "残基负例")
    atom_pos = load_feat(data_paths["atom_pos_train"], "原子正例")
    atom_neg = load_feat(data_paths["atom_neg_train"], "原子负例")

    if not all([len(res_pos) == len(atom_pos), len(res_neg) == len(atom_neg)]):
        logger.error("正/负例残基和原子特征数量不匹配！")
        return

    ALL_RES_FEATS = list(res_pos) + list(res_neg)
    ALL_ATOM_FEATS = list(atom_pos) + list(atom_neg)
    ALL_LABELS = [1] * len(res_pos) + [0] * len(res_neg)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    all_fold_best_acc = []

    for fold_num, (train_idx, val_idx) in enumerate(kf.split(ALL_RES_FEATS), 1):
        logger.info(f"\n{'=' * 50} 开始第 {fold_num}/5 折训练 {'=' * 50}")

        train_res = [ALL_RES_FEATS[i] for i in train_idx]
        train_atom = [ALL_ATOM_FEATS[i] for i in train_idx]
        train_labels = [ALL_LABELS[i] for i in train_idx]

        val_res = [ALL_RES_FEATS[i] for i in val_idx]
        val_atom = [ALL_ATOM_FEATS[i] for i in val_idx]
        val_labels = [ALL_LABELS[i] for i in val_idx]

        train_dataset = DualChannelDataset(train_res, train_atom, train_labels, config["expected_res_dim"],
                                           config["expected_atom_dim"])
        val_dataset = DualChannelDataset(val_res, val_atom, val_labels, config["expected_res_dim"],
                                         config["expected_atom_dim"])

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                                  collate_fn=dual_collate_fn, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=dual_collate_fn,
                                num_workers=0)

        model = DualChannelAMPClassifier(
            res_dim=config["expected_res_dim"],
            atom_dim=config["expected_atom_dim"],
            hidden_dim=config["hidden_dim"],
            num_heads=config["num_heads"],
            dropout_rate=config["dropout_rate"]
        ).to(config["device"])

        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scaler = GradScaler('cuda', enabled=(config["device"].type == 'cuda'))



        best_acc = 0.0
        early_stopping_counter = 0
        os.makedirs("32d", exist_ok=True)

        for epoch in range(1, config["epochs"] + 1):
            logger.info(f"折{fold_num} | Epoch {epoch}/{config['epochs']} | LR: {optimizer.param_groups[0]['lr']:.8f}")
            train_loss = train(model, train_loader, optimizer, criterion, config["device"], scaler)
            val_metrics = evaluate(model, val_loader, config["device"], config["threshold"])



            if val_metrics["acc"] > best_acc:
                best_acc = val_metrics["acc"]
                torch.save(model.state_dict(), f"32d/shiyan_fold{fold_num}_best.pth")
                logger.info(f"折{fold_num} 保存最优模型: ACC {best_acc:.4f}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= config["early_stopping_patience"]:
                logger.info(f"折{fold_num} 早停触发！最佳验证集ACC: {best_acc:.4f}")
                break

        all_fold_best_acc.append(best_acc)
        logger.info(f"\n{'=' * 50} 第 {fold_num}/5 折训练结束，最优ACC: {best_acc:.4f} {'=' * 50}")

    logger.info("\n" + "=" * 70)
    logger.info(f"五折平均ACC：{np.mean(all_fold_best_acc):.4f} ± {np.std(all_fold_best_acc):.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()