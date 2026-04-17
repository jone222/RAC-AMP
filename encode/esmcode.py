import torch
import esm
from esm.pretrained import esm2_t33_650M_UR50D
import os
import pickle
from Bio import SeqIO
import numpy as np


def generate_esm_embeddings(
        fasta_path,
        output_dir,
        save_cls=True,
        save_residues=False,
        batch_size=8
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"批次大小: {batch_size}")

    model, alphabet = esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    all_cls_features = []
    all_sequence_ids = []
    all_residue_features = []

    print(f"读取序列文件: {fasta_path}")
    with open(fasta_path, "r", encoding="utf-8") as f:
        records = list(SeqIO.parse(f, "fasta"))
    total = len(records)
    print(f"共发现 {total} 条序列，开始批量处理...")

    batch_converter = alphabet.get_batch_converter()
    for i in range(0, total, batch_size):
        batch_records = records[i:i + batch_size]
        batch_ids = [rec.id for rec in batch_records]
        batch_seqs = [str(rec.seq).upper() for rec in batch_records]

        print(f"\n处理批次 {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")
        print(f"包含序列: {', '.join(batch_ids)}")

        data = list(zip(batch_ids, batch_seqs))
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)


        batch_real_lengths = [len(seq) + 2 for seq in batch_seqs]

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_embeddings = results["representations"][33].cpu().numpy()

        for idx in range(len(batch_records)):
            seq_id = batch_ids[idx]


            real_embeddings = token_embeddings[idx, :batch_real_lengths[idx]]


            embeddings_without_eos = real_embeddings[:-1]

            all_sequence_ids.append(seq_id)

            if save_cls:
                cls_feature = embeddings_without_eos[0:1, :]  # 提取<cls>
                all_cls_features.append(cls_feature)

            if save_residues:
                residue_features = embeddings_without_eos[1:, :]  # 去除<cls>

        print(f"批次 {i // batch_size + 1} 处理完成")


    if save_cls:
        cls_array = np.concatenate(all_cls_features, axis=0)
        cls_save_path = os.path.join(output_dir, "all_cls_features.npy")
        np.save(cls_save_path, cls_array)
        print(f"所有cls特征已保存到: {cls_save_path}")

    if save_residues:
        residues_save_path = os.path.join(output_dir, "all_residue_features.pkl")
        with open(residues_save_path, "wb") as f:
            pickle.dump(all_residue_features, f)
        print(f"所有残基特征已保存到: {residues_save_path}")

    ids_save_path = os.path.join(output_dir, "sequence_ids.npy")
    np.save(ids_save_path, all_sequence_ids)
    print(f"序列ID列表已保存到: {ids_save_path}")

    print("\n所有序列处理完成")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    FASTA_PATH ="" # 你的FASTA路径
    OUTPUT_DIR =""  # 输出目录

    generate_esm_embeddings(
        fasta_path=FASTA_PATH,
        output_dir=OUTPUT_DIR,
        save_cls=True,
        save_residues=True,
        batch_size=32
    )
