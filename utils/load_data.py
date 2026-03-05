import numpy as np
from utils.encoding_methods import onehot_encoding, pssm_encoding
import os
from utils.util_methods import *
import pandas as pd
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import torch.nn as nn


padding_len = 50
batch_size = 32 

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()  # 使用 ReLU 激活函数
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)  # 激活函数
        x = self.fc2(x)
        return x

def load_seqs_with_labels(folder_fasta: str,  *fs : str):
    """
    :param folder_fasta: fasta文件所在目录
    :param fs: fasta文件名，list[str]
    """
    if folder_fasta[-1] != '/': folder_fasta += '/'
    seq2class = {}  # 将序列映射到类别
    n_class = len(fs)

    for i, fn in enumerate(fs):
        ids, seqs = fasta_parser(folder_fasta + fn)
        seqs = set(seqs)

        for seq in seqs:
            if seq in seq2class.keys():
                seq2class[seq][i] = 1
            else:
                seq2class[seq] = np.zeros(n_class)
                seq2class[seq][i] = 1

    return list(seq2class.keys()), np.array(list(seq2class.values()), dtype=np.int)

def load_seqs_and_labels(folder_fasta, names):

    ids, seqs = fasta_parser(os.path.join(folder_fasta, "seqs.fasta"))
    df = pd.read_csv(os.path.join(folder_fasta, "labels.csv"))
    df = df[names]
    labels = np.array(df.values, dtype=np.int)

    return seqs, labels

def pad_by_zero(x, max_len):
    padded_encodings = []
    masks = []      # (n_samples, len), mask = True if the position is padded by zero

    for sample in x:
        # sample: (len, fea_dim)
        if sample.shape[0] < padding_len:
            pad_zeros = np.zeros((padding_len - sample.shape[0], sample.shape[1]), dtype=np.int)
            padded_enc = np.vstack((sample, pad_zeros))
            padded_encodings.append(padded_enc)

            pad_mask = np.ones((padding_len - sample.shape[0]), dtype=np.int)
            non_mask = np.zeros((sample.shape[0]))
            msk = np.hstack((non_mask, pad_mask)) == 1
            masks.append(msk)
        else:
            # >=  padding_len
            tsample = np.vstack((sample[:padding_len // 2, :], sample[-padding_len // 2:, :]))
            padded_encodings.append(tsample)
            non_mask = np.zeros((padding_len))
            masks.append(non_mask == 1)
        # else:
        #     # >=  padding_len
        #     # tsample = np.vstack((sample[:padding_len // 2, :], sample[-padding_len // 2:, :]))
        #     tsample = sample[:padding_len, :]
        #     padded_encodings.append(tsample)
        #     non_mask = np.zeros((padding_len))
        #     masks.append(non_mask == 1)

    res = np.array(padded_encodings)   # n_samples, len, fea_dim
    masks = np.array(masks)

    return res, masks

def encode_protein_sequences(seqs, batch_size, max_length=50, device=None):
    """
    ProtT5-XL
    """
    if device is None:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model_path = "/disk1/home/lusiyan/TPpred-PepPA/pretrain_model/prot_t5"
    
    
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)

    model = T5EncoderModel.from_pretrained(model_path).to(device)

    if device.type == "cpu":
        model.to(torch.float32)

    processed_seqs = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in seqs]

    all_embeddings = []

    for i in range(0, len(processed_seqs), batch_size):
        batch_sequences = processed_seqs[i:i + batch_size]
        
        ids = tokenizer(batch_sequences, add_special_tokens=True, padding="max_length", 
                        max_length=max_length, truncation=True, return_tensors="pt")

        input_ids = ids['input_ids'].to(device)
        attention_mask = ids['attention_mask'].to(device)

        with torch.no_grad():
            batch_embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        
        all_embeddings.append(batch_embedding.last_hidden_state.cpu().numpy())
   
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    return all_embeddings



def load_features(folder_fasta: str, padding : bool, *fs : str, seed=42):

    names = [x[:-4] for x in fs]
    seqs, labels = load_seqs_and_labels(folder_fasta, names)
    onehot_enc = onehot_encoding(seqs)   #（N,L_i,20）
    pssm_enc = pssm_encoding(seqs, 'features/pssm/', True)  
    
    print("onehot_enc length:", len(onehot_enc))
    print("pssm_enc length:", len(pssm_enc))

    masks = [] # (n_samples, len)
   
    res = cat(onehot_enc, pssm_enc)  

    if padding:
        res, masks = pad_by_zero(res, padding_len)  
    print("res1",res.shape)
    print("mask1",masks.shape)

    embeddings = encode_protein_sequences(seqs, batch_size)   
    print("embeddings",embeddings.shape)
    print("0")

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32) 

    input_dim = 1024  
    hidden_dim = 512  
    output_dim = 20
    
    cpu_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        gpu_rng_state = torch.cuda.get_rng_state()
   
    torch.manual_seed(seed)           
    
    mlp = MLP(input_dim, hidden_dim, output_dim)
    mlp.eval() 
    
    torch.set_rng_state(cpu_rng_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(gpu_rng_state)

    with torch.no_grad():
        compressed_embeddings = mlp(embeddings_tensor)

    compressed_embeddings = compressed_embeddings.detach().numpy()
     
    result = np.concatenate((compressed_embeddings, res), axis=2)

    return result,labels, masks, seqs




