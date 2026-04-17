import numpy as np
import pickle
import os
import warnings
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import ValenceType
from typing import List, Dict, Tuple, Any


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('equivariant_conversion.log'), logging.StreamHandler()]
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


aa_dict = {
    'a': 'C[C@H]([NH2:1])[C:1](=[O:1])[O:1]',
    'c': '[NH2:1][C@@H](CS)[C:1](=[O:1])[O:1]',
    'd': '[NH2:1][C@@H](CC(=O)O)[C:1](=[O:1])[OH:1]',
    'e': '[NH2:1][C@@H](CCC(=O)O)[C:1](=[O:1])[OH:1]',
    'f': '[NH2:1][C@@H](Cc1ccccc1)[C:1](=[O:1])[OH:1]',
    'g': '[NH2:1]C[C:1](=[O:1])[OH:1]',
    'h': '[NH2:1][C@@H](Cc1c[nH]cn1)[C:1](=[O:1])[OH:1]',
    'i': 'CC[C@H](C)[C@H]([NH2:1])[C:1](=[O:1])[OH:1]',
    'k': 'NCCCC[C@H]([NH2:1])[C:1](=[O:1])[OH:1]',
    'l': 'CC(C)C[C@H]([NH2:1])[C:1](=[O:1])[OH:1]',
    'm': 'CSCC[C@H]([NH2:1])[C:1](=[O:1])[OH:1]',
    'n': 'NC(=O)C[C@H]([NH2:1])[C:1](=[O:1])[OH:1]',
    'p': '[O:1]=[C:1]([OH:1])[C@@H]1CCC[NH:1]1',
    'q': 'NC(=O)CC[C@H]([NH2:1])[C:1](=[O:1])[OH:1]',
    'r': 'N=C(N)NCCC[C@H]([NH2:1])[C:1](=[O:1])[OH:1]',
    's': '[NH2:1][C@@H](CO)[C:1](=[O:1])[OH:1]',
    't': 'C[C@@H](O)[C@H]([NH2:1])[C:1](=[O:1])[OH:1]',
    'v': 'CC(C)[C@H]([NH2:1])[C:1](=[O:1])[OH:1]',
    'w': '[NH2:1][C@@H](Cc1c[nH]c2ccccc12)[C:1](=[O:1])[OH:1]',
    'y': '[NH2:1][C@@H](Cc1ccc(O)cc1)[C:1](=[O:1])[OH:1]'
}


pre_known_charges = {
    'a': [-0.038941544836773818, 0.10498306785206031, -0.31852489852720955, 0.37128096695769647, -0.245609700240111,
          -0.245609700240111],
    'c': [-0.31760762075618459, 0.11671109552669053, 0.016444904280613366, -0.17711583286998214, 0.37240697866736511,
          -0.24556022418544537, -0.24556022418544537],
    'd': [-0.31795147828443121, 0.11365865612305733, 0.071241260785456353, 0.30503469709389208, -0.2523968033036173,
          -0.48117538323698905, 0.32057810707028217, -0.25058130351573815, -0.48007458571726869],
    'e': [-0.3184276525520085, 0.10368324144471773, -0.016649349421885624, 0.047372069789650653, 0.30291723427615275,
          -0.25248442381067765, -0.48122949062943815, 0.31998403007662279, -0.2506002043614457, -0.48008627987892583],
    'f': [-0.31816209755519703, 0.10718797936647743, -0.0017896024515534755, -0.045878604848919036,
          -0.058969174493864206, -0.06199304993033751, -0.06224871979673649, -0.06199304993033751,
          -0.058969174493864206, 0.32031550163956013, -0.25058631033747231, -0.48007770020344237],
    'g': [-0.32100388796548518, 0.091814239632494035, 0.31675597840558584, -0.25094762709006846, -0.48029713002439572],
    'h': [-0.31811218613827036, 0.10876025651267705, 0.017178897164274905, 0.060240393418915093, 0.023612279812735563,
          -0.35090363274243741, 0.092256578126356181, -0.24194850421171329, 0.32037773191321722, -0.25058513638855856,
          -0.48007696992659998],
    'i': [-0.065054342792623523, -0.051516109532843919, -0.01848141138509252, -0.060455687487424026, 0.1056921548136058,
          -0.31821728010260475, 0.3202466808084985, -0.25058798571573865, -0.48007873928516198],
    'k': [-0.33047349405550447, -0.0077263173078670703, -0.040819171235958823, -0.05005067194555176,
          -0.027103291386504948, 0.10312285864403094, -0.31844107224794488, 0.31996730024702658, -0.2506004737905258,
          -0.480086447570925],
    'l': [-0.062699721671212155, -0.045080166422972974, -0.062699721671212155, -0.024825714877267389,
          0.10336652779889306, -0.31843253370427399, 0.31997794662058598, -0.25060026439075556, -0.48008631732626356],
    'm': [-0.018398913027632899, -0.1653569719042017, -0.00497659329816642, -0.018028983910199952, 0.10391068699378787,
          -0.31841548940258046, 0.31999919762538026, -0.25059986574145887, -0.48008606933473558],
    'n': [-0.3696138646860766, 0.21908715699891673, -0.27528806213181317, 0.052212076976201664, 0.11193696290241258,
          -0.31800778429812587, 0.32050790323073358, -0.25058264416049209, -0.48007541965947526],
    'p': [-0.25059127541388582, 0.32019187971860097, -0.48008076676695011, 0.1056602413452867, -0.025726752171922784,
          -0.038583656372423773, -0.0040133371139500825, -0.30441260474699489],
    'q': [-0.36967889475861154, 0.21698956994421906, -0.27537278910866281, 0.028297174649971519, -0.018362474406193877,
          0.10361820954695432, -0.31842860380192411, 0.31998284457023513, -0.25060021491233908, -0.48008628646499379],
    'r': [-0.26962207123355619, 0.18520502814523329, -0.37016043799114567, -0.35676207127350623, 0.017531796964524702,
          -0.03363639331536647, -0.025476863927366322, 0.10318774485000681, -0.31844010091909719, 0.31996851078193783,
          -0.25060046290166194, -0.48008644077399965],
    's': [-0.31651542552659373, 0.12646568376767636, 0.069022807911391801, -0.39413557675578653, 0.32237114684868762,
          -0.25049509689415916, -0.4800213961433728],
    't': [-0.036810537847947578, 0.077045484390140126, -0.39117403578585813, 0.12904528397618095, -0.31628111187635122,
          0.32266366773949673, -0.25048182094826726, -0.4800132023099517],
    'v': [-0.060711309783474096, -0.021064293477971893, -0.060711309783474096, 0.10543802912336016,
          -0.31822595394937808, 0.32023586583935859, -0.25058819622173711, -0.48007887022192064],
    'w': [-0.31816095861050087, 0.10726664689345959, 0.00032329871948459876, -0.020502848541995324,
          0.0049744588987986468, -0.36088879768036697, 0.045629818510211806, -0.038003021926457513,
          -0.060158911668000872, -0.061509233239626836, -0.052562691519528068, 0.0026344222033379563,
          0.32031692105916915, -0.25058629777133862, -0.48007769235935871],
    'y': [-0.31816209743565643, 0.10718801610357671, -0.0017850394584956058, -0.045610041081834043,
          -0.055294427338870203, -0.019954155177891012, 0.11509485950992565, -0.50796664517588408,
          -0.019954155177891012, -0.055294427338870203, 0.32031550178849594, -0.25058631033716872,
          -0.0017850394584956058]
}


def one_of_k_encoding_unk(x, allowable_set):
    """One-hot编码（未知值映射到最后一类）"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def get_ring_info(atom):
    """获取原子环信息（5元环/6元环）"""
    return [1 if atom.IsInRingSize(i) else 0 for i in range(5, 7)]


def atom_feature(atom) -> np.ndarray:
    """提取单个原子的特征（20维），兼容新旧版本RDKit的价态获取方式"""
    try:
        symbol = atom.GetSymbol()


        try:
            explicit_valence = atom.GetValence(which=ValenceType.Explicit)
        except AttributeError:
            explicit_valence = atom.GetExplicitValence()

        return np.array(
            # 元素类型（5维：C, N, O, S, H）
            one_of_k_encoding_unk(symbol, ['C', 'N', 'O', 'S', 'H']) +
            # 连接度（4维：0-3）
            one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3]) +
            # 氢原子数（3维：0-2）
            one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2]) +
            # 显式价态（3维：0-2）
            one_of_k_encoding_unk(explicit_valence, [0, 1, 2]) +
            # 是否芳香族（1维）
            [1.0 if atom.GetIsAromatic() else 0.0] +
            # 环信息（2维：5/6元环）
            get_ring_info(atom) +
            # 杂化方式（2维：sp, sp2, sp3, 其他 → 取前2维）
            one_of_k_encoding_unk(int(atom.GetHybridization()), [1, 2, 3])[:2]
        ).astype(np.float32)
    except Exception as e:
        logging.error(f"提取原子特征错误: {e}（原子符号: {atom.GetSymbol()}）")
        return np.zeros(20, dtype=np.float32)  # 确保返回20维


def generate_3d_coordinates(mol: Chem.Mol) -> Tuple[np.ndarray, Chem.Mol]:
    """生成3D坐标（含氢原子）"""
    try:
        mol_withH = Chem.AddHs(mol)  # 强制加氢
        status = AllChem.EmbedMolecule(mol_withH, AllChem.ETKDGv2())
        if status != 0:
            logging.warning("3D构象生成失败，使用随机构象")
            AllChem.EmbedMolecule(mol_withH, randomSeed=42)

        try:
            AllChem.MMFFOptimizeMolecule(mol_withH)
        except Exception as e:
            logging.warning(f"构象优化失败: {e}，使用初始构象")

        conf = mol_withH.GetConformer()
        coords = np.array([
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol_withH.GetNumAtoms())
        ], dtype=np.float32)

        return coords, mol_withH

    except Exception as e:
        logging.error(f"生成3D坐标错误: {e}")
        mol_withH = Chem.AddHs(mol)
        num_atoms = mol_withH.GetNumAtoms()
        return np.zeros((num_atoms, 3), dtype=np.float32), mol_withH


def get_edge_index(adj_matrix: np.ndarray) -> np.ndarray:
    """将邻接矩阵转换为边索引（2, num_edges）"""
    edges = np.where(adj_matrix > 0)
    return np.stack(edges, axis=0)


def convert_to_equivariant_representation(seq: str) -> List[Dict[str, Any]]:
    """生成等变图神经网络表示"""
    seq = seq.lower()
    equivariant_reps = []

    for idx, aa in enumerate(seq):
        try:
            if aa not in aa_dict:
                logging.warning(f"位置 {idx} 跳过不支持的氨基酸: {aa}")
                continue

            smiles = aa_dict[aa]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logging.error(f"氨基酸 {aa} 的SMILES解析失败: {smiles}")
                continue

            # 生成坐标和加氢分子
            coords, mol_withH = generate_3d_coordinates(mol)
            num_atoms = mol_withH.GetNumAtoms()

            # 边索引
            adj = Chem.GetAdjacencyMatrix(mol_withH)
            edge_index = get_edge_index(adj)

            # 原子特征（确保20维）
            atom_features = []
            for atom in mol_withH.GetAtoms():
                feat = atom_feature(atom)
                atom_features.append(feat)
            atom_features_array = np.array(atom_features, dtype=np.float32)
            if atom_features_array.shape[1] != 20:
                logging.error(f"氨基酸 {aa} 特征维度错误: {atom_features_array.shape[1]}（预期20）")
                continue

            # 特征归一化
            atom_features_norm = atom_features_array / (
                    atom_features_array.sum(axis=1, keepdims=True) + 1e-8
            )

            # 节点特征（20维结构特征 + 1维电荷 → 21维）
            node_features = np.zeros((num_atoms, 21), dtype=np.float32)
            node_features[:, :20] = atom_features_norm

            # 获取非氢原子索引（核心修改：只给非氢原子分配预定义电荷）
            non_h_indices = [i for i, atom in enumerate(mol_withH.GetAtoms()) if atom.GetSymbol() != 'H']
            num_non_h = len(non_h_indices)

            # 填充电荷（修改后逻辑）
            if aa in pre_known_charges:
                pre_charges = pre_known_charges[aa]
                if len(pre_charges) == num_non_h:
                    # 初始化所有电荷为0（氢原子保持0）
                    node_features[:, 20] = 0.0
                    # 仅为非氢原子分配预定义电荷
                    for i, idx in enumerate(non_h_indices):
                        node_features[idx, 20] = pre_charges[i]
                    logging.debug(f"氨基酸 {aa} 非氢原子电荷分配完成（非氢原子数: {num_non_h}）")
                else:
                    logging.warning(
                        f"氨基酸 {aa} 非氢原子数({num_non_h})与预定义电荷数({len(pre_charges)})不匹配，使用0电荷"
                    )
                    node_features[:, 20] = 0.0
            else:
                node_features[:, 20] = 0.0
                logging.warning(f"氨基酸 {aa} 无预定义电荷，使用0电荷")

            # 掩码
            mask = np.ones(num_atoms, dtype=np.float32)

            equivariant_reps.append({
                'node_features': node_features,
                'positions': coords,
                'edge_index': edge_index,
                'mask': mask,
                'aa': aa
            })

        except Exception as e:
            logging.error(f"处理氨基酸 {aa}（位置 {idx}）时出错: {e}")
            continue

    return equivariant_reps


def read_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    sequences = []
    seen_ids = set()
    duplicate_count = 0

    try:
        if not os.path.exists(fasta_path):
            logging.error(f"FASTA文件不存在: {fasta_path}")
            return []

        with open(fasta_path, 'r', encoding='utf-8') as f:
            current_id = None
            current_seq = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                if line.startswith('>'):
                    if current_id is not None and current_seq:
                        seq_str = ''.join(current_seq).replace(' ', '').upper()
                        if current_id in seen_ids:
                            duplicate_count += 1
                            unique_id = f"{current_id}_dup{duplicate_count}"
                        else:
                            unique_id = current_id
                            seen_ids.add(current_id)
                        sequences.append((unique_id, seq_str))
                    current_seq = []
                    current_id = line[1:].split()[0] if ' ' in line else line[1:]
                    if not current_id:
                        current_id = f"unknown_id_{line_num}"
                        logging.warning(f"行 {line_num} 未找到有效ID，使用默认ID: {current_id}")

                else:
                    current_seq.append(line)

            if current_id is not None and current_seq:
                seq_str = ''.join(current_seq).replace(' ', '').upper()
                if current_id in seen_ids:
                    duplicate_count += 1
                    unique_id = f"{current_id}_dup{duplicate_count}"
                else:
                    unique_id = current_id
                sequences.append((unique_id, seq_str))

        logging.info(f"FASTA文件读取完成，总序列数: {len(sequences)}，重复ID处理数: {duplicate_count}")

    except Exception as e:
        logging.error(f"读取FASTA文件错误: {e}")

    return sequences


def fasta_to_equivariant_representation(fasta_path: str, output_dir: str = None) -> Dict[str, List[Dict[str, Any]]]:
    logging.info(f"开始处理FASTA文件: {fasta_path}")
    sequences = read_fasta(fasta_path)

    if not sequences:
        logging.error("未读取到任何有效序列")
        return {}

    seq_lengths = [len(seq) for _, seq in sequences]
    logging.info(
        f"序列统计: 总数={len(sequences)}, "
        f"长度范围=[{min(seq_lengths)},{max(seq_lengths)}], "
        f"平均长度={np.mean(seq_lengths):.2f}"
    )

    representations = {}
    processed_count = 0
    success_count = 0

    for seq_id, seq in sequences:
        processed_count += 1
        try:
            rep = convert_to_equivariant_representation(seq)
            if rep:
                representations[seq_id] = rep
                success_count += 1
            else:
                logging.warning(f"序列 {seq_id} 未生成有效表示")

            if processed_count % 100 == 0:
                logging.info(f"进度: 已处理 {processed_count}/{len(sequences)}，成功 {success_count}")

        except Exception as e:
            logging.error(f"处理序列 {seq_id} 失败: {e}")
            continue

    logging.info("\n处理完成统计:")
    logging.info(f"- 总序列数: {len(sequences)}")
    logging.info(f"- 成功处理: {success_count}")
    logging.info(f"- 失败: {len(sequences) - success_count}")
    logging.info(f"- 成功率: {success_count / len(sequences) * 100:.2f}%")

    if output_dir and representations:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "k1.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(representations, f)
        logging.info(f"等变表示已保存到: {output_path}（{len(representations)}个序列）")
    elif not representations:
        logging.warning("无有效表示可保存")

    return representations


if __name__ == "__main__":
    input_fasta = r""# 替换为你的FASTA文件路径
    output_directory = ""

    representations = fasta_to_equivariant_representation(input_fasta, output_directory)

    if representations:
        first_id = next(iter(representations.keys()))
        first_rep = representations[first_id]
        logging.info(f"\n示例: 序列 {first_id}")
        logging.info(f"- 氨基酸个数: {len(first_rep)}")

        if first_rep:
            first_aa = first_rep[0]
            logging.info(f"- 第一个氨基酸({first_aa['aa']})的原子数: {first_aa['node_features'].shape[0]}")
            logging.info(f"- 节点特征形状: {first_aa['node_features'].shape}")
            logging.info(f"- 坐标形状: {first_aa['positions'].shape}")
            logging.info(f"- 边索引形状: {first_aa['edge_index'].shape}")
            logging.info(f"- 掩码形状: {first_aa['mask'].shape}")
    else:
        logging.info("未生成任何有效表示")