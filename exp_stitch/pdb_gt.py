#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.optim as optim
import torch_geometric
from torch.nn.functional import relu, sigmoid, softmax, mse_loss
from torch.nn import Linear, Module, Dropout, MSELoss, CrossEntropyLoss, BatchNorm1d

from tqdm import tqdm

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_mean_pool, TransformerConv

import pandas as pd
import numpy as np

import os
import pickle
import gzip
import optuna

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
device = 0
device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")


# In[2]:


class MultiHeadAttention(Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0

        self.query_linear = Linear(hidden_dim, hidden_dim)
        self.key_linear = Linear(hidden_dim, hidden_dim)
        self.value_linear = Linear(hidden_dim, hidden_dim)

        self.output_linear = Linear(hidden_dim, hidden_dim)
        self.dropout = Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim // num_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        Q = Q.view(batch_size, self.num_heads, -1, self.hidden_dim // self.num_heads)
        K = K.view(batch_size, self.num_heads, -1, self.hidden_dim // self.num_heads)
        V = V.view(batch_size, self.num_heads, -1, self.hidden_dim // self.num_heads)
    
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.dropout(softmax(energy, dim=-1))

        weighted_matrix = torch.matmul(attention, V)

        weighted_matrix = weighted_matrix.permute(0, 2, 1, 3).contiguous()
        weighted_matrix = weighted_matrix.view(batch_size, -1, self.hidden_dim)
        weighted_matrix = weighted_matrix[:, 0, :]

        output = self.output_linear(weighted_matrix)

        return output


# In[3]:


class DrugEncoder(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden, num_emb):
        super(DrugEncoder, self).__init__()
        self.conv1 = TransformerConv(num_node_features, num_hidden)
        self.conv2 = TransformerConv(num_hidden, num_emb)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = x.float()

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return x


# In[4]:


class ProteinEncoder(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden, num_emb):
        super(ProteinEncoder, self).__init__()
        self.conv1 = TransformerConv(num_node_features, num_hidden)
        self.conv2 = TransformerConv(num_hidden, num_emb)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return x


# In[5]:


class DTIPredictor(Module):
    def __init__(
        self, drug_encoder, protein_encoder, drug_hidden=16, protein_hidden=16, 
        emb_dim=100, num_heads=4, dropout=0.2, is_attention=True, is_cmap=True
    ):
        super().__init__()
        self.drug_encoder = DrugEncoder(9, drug_hidden, emb_dim)
        self.protein_encoder = ProteinEncoder(1280 if is_cmap else 5, protein_hidden, emb_dim)
        self.is_attention = is_attention
        
        if is_attention:
            self.attention = MultiHeadAttention(emb_dim, num_heads, dropout)
            self.fc_output = Linear(emb_dim*3, 1)
        else:
            self.fc_output = Linear(emb_dim*2, 1)
            
    def forward(self, drug_data, protein_data):
        x_drug = self.drug_encoder(drug_data)
        x_protein = self.protein_encoder(protein_data)
        
        if self.is_attention:
            attention_output = self.attention(x_drug, x_protein, x_protein)

            feature = torch.cat((x_drug, attention_output, x_protein), dim=1)
            prediction = self.fc_output(feature)
        else:
            feature = torch.cat((x_drug, x_protein), dim=1)
            prediction = self.fc_output(feature)
            
        return (prediction).squeeze(1)


# In[ ]:





# In[6]:


train = pd.read_csv('../stitch/train.csv', index_col=0)
val = pd.read_csv('../stitch/val.csv', index_col=0)
test = pd.read_csv('../stitch/test.csv', index_col=0)


# In[7]:


print('Train dim:', train.shape)
print('val dim:', val.shape)
print('test dim:', test.shape)


# In[8]:


with gzip.open('../drug.pkl.gz', 'rb') as f:
    drug = pickle.load(f)

def get_drug_dataloader(drugs, batch_size=100):
    dataset = [drug[i] for i in drugs]
    return DataLoader(dataset, batch_size=batch_size)

def get_protein_dataloader(proteins, batch_size=100, is_cmap=True):
    if is_cmap:
        dataset = [torch.load('../cmap/{}.pt'.format(i)) for i in proteins]
    else:
        dataset = [torch.load('../protein_graphs/{}.pt'.format(i)) for i in proteins]
        
    return DataLoader(dataset, batch_size=batch_size)


# In[ ]:





# In[18]:


def objective(trial):
#     batch_size, drug_hidden, protein_hidden, lr, epochs
#     emb_dim, num_heads, dropout, is_attention, is_cmap
    
    batch_size = trial.suggest_categorical("batch_size", [50, 100, 150])
    
    drug_train_loader = get_drug_dataloader(train['Drug'], batch_size)
    drug_val_loader = get_drug_dataloader(val['Drug'], batch_size)
    drug_test_loader = get_drug_dataloader(test['Drug'], batch_size)
    
#     is_cmap = trial.suggest_categorical("is_cmap", [True, False])
    is_cmap = False
    
    protein_train_loader = get_protein_dataloader(train['Entry'], batch_size, is_cmap=is_cmap)
    protein_val_loader = get_protein_dataloader(val['Entry'], batch_size, is_cmap=is_cmap)
    protein_test_loader = get_protein_dataloader(test['Entry'], batch_size, is_cmap=is_cmap)
    
    train_y = DataLoader(torch.Tensor(train['combined_score'].values).float(), batch_size=batch_size)
    val_y = DataLoader(torch.Tensor(val['combined_score'].values).float(), batch_size=batch_size)
    test_y = DataLoader(torch.Tensor(test['combined_score'].values).float(), batch_size=batch_size)

    drug_hidden = trial.suggest_categorical("drug_hidden", [16, 32, 64])
    protein_hidden = trial.suggest_categorical("protein_hidden", [16, 32, 64])
    emb_dim = 100
    num_heads = 4
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])
    is_attention = trial.suggest_categorical("is_attention", [True, False])
    
    model = DTIPredictor(
        DrugEncoder, ProteinEncoder, 
        drug_hidden, protein_hidden, emb_dim, 
        num_heads, dropout, is_attention, is_cmap
    ).to(device)
    criterion = MSELoss().to(device)
    
    lr = trial.suggest_categorical("lr", [0.01, 0.001, 0.0001])
    
    optimizer = getattr(torch.optim, "Adam")(model.parameters(), lr=0.01,)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    epochs = trial.suggest_categorical("epochs", [10, 50, 100])
    for epoch in (range(epochs)):

        model.train()
        total_loss = 0
        for drug, protein, true_y in zip(drug_train_loader, protein_train_loader, train_y):
            drug = drug.to(device)
            protein = protein.to(device)
            true_y = true_y.to(device)

            optimizer.zero_grad()

            output = model(drug, protein)
            loss = criterion(output, true_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(drug_train_loader)
        train_losses.append(average_loss)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for drug, protein, true_y in zip(drug_val_loader, protein_val_loader, val_y):
                drug = drug.to(device)
                protein = protein.to(device)
                true_y = true_y.to(device)            

                output = model(drug, protein)
                loss = criterion(output, true_y)
                val_loss += loss.item()
                val_losses.append(val_loss)

            average_val_loss = val_loss / len(drug_val_loader)
            val_losses.append(average_val_loss)
    
    return average_loss, average_val_loss


# In[19]:


study = optuna.create_study(
    directions=["minimize"]*2,
    storage="sqlite:///TransformerConv.sqlite3",
    study_name="TransformerConv",
    load_if_exists=True,
)
study.optimize(objective, n_trials=1000, gc_after_trial=True)


# In[20]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




