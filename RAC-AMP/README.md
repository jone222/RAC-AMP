# RAC-AMP: A Dual-Channel Framework for Antimicrobial Peptide Prediction via Residue-Atom Cross-Attention

## Introduction
![Framework](1.svg) (a) Residue Channel: Extracts evolutionary sequence features via the pre-trained ESM-2 model and generates the global residue representation through self-attention and pooling. (b) Atom Channel: Constructs residue-level atomic graphs with 3D coordinates as geometry priors and employs EGNN\citep{satorras2021n} to capture rotation- and translation-invariant atomic features for the global atom representation. (c) Global Cross-Attention and Classification: Fuses sequence and structural features via cross-attention, concatenates the fused features with the original residue features, and predicts AMP probability using fully connected layers and a sigmoid classifier. (d) Cross-Modality Interaction Mechanism: A multi-head cross-attention block that achieves fine-grained fusion between residue and atom features to form a robust hybrid representation for AMP identification.

## Environment Setup & Installation

This project is developed and evaluated using **Python 3.13** and **PyTorch 2.8.0** with **CUDA 12.9** support. 

To set up the environment and install all necessary dependencies, please run the following command:

```bash
pip install -r requirements.txt
```
## Data Preprocessing
Before training or inference, raw FASTA sequences must be encoded into residue-level and atom-level features to satisfy the dual-channel input requirements.

### 1. Residue Encoding
Use `esmcode.py` to extract semantic embeddings from protein sequences (based on the ESM-2 model).
### 2. Atom Encoding
Use `st.py` to process structural information and generate atomic-level spatial features and coordinate graphs.
## Training
Once the data preprocessing is complete, you can start the training process.
```bash
python train.py
```

## Evaluation
To evaluate the performance of the trained **RAC-AMP** model on an independent test set or to perform inference on new peptide data, run:

```bash
python test.py
```
## Citation
If you find **RAC-AMP** helpful in your research, please consider citing our paper.