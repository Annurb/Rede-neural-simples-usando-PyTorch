# Rede neural com um perceptron de representação de uma reta
import torch
import numpy as np
from torch import nn


class LineNetwork(nn.Module):
    # Inicialização
    def __init__(self):
        # Inicialização da superclasse
        super().__init__()
        # Camada
        self.layers = nn.Sequential(
            # Recebe uma entrada(neuronio) e da um valor
            nn.Linear(1, 1)
        )

    # Como a rede computa, recebe um dado e passa pelo perceptron
    def forward(self, x):
        return self.layers(x)

from torch.utils.data import Dataset, DataLoader
import torch.distributions.uniform as urand

# Criação de um dataset
class AlgebraicDataset(Dataset):
    # Função, intervalo da função, quantidade de pontos
    def __init__(self, f, interval, nsamples):
        # Mostrando pontos uniformemente entre o inicio do intervalo e o final e a amostragem da quantidade de pontos passada
        X = urand.Uniform(interval[0], interval[1]).sample([nsamples])
        # Conjunto de dados como uma lista de pares ordenados
        self.data = [(x, f(x)) for x in X]

    # Quantos dados tem no dataset
    def __len__(self):
         # Tamanho do conjunto de dados como o tamanho da lista
        return len(self.data)
    # Qual dado ter, quando passar o indice do dado
    def __getitem__(self, idx):
        return self.data[idx]

# Linha, intervalo, qnt de amostras de treinamento, qnt de testes    
line = lambda x: 2*x+3
interval = (-10, 10)
train_nsamples = 1000
test_nsamples = 100

# Parâmetros passados para a classe
train_dataset = AlgebraicDataset(line, interval, train_nsamples)
test_dataset = AlgebraicDataset(line, interval, test_nsamples)

# Quem vai me dar os dados ao longo do treinamento
# Dataset de treinamento, olhar todos os dados(rede simples) e embaralhamento de dados
# Quando é uma rede complexa com mais dados nao lemos os dados todos de uma vez
train_dataloader = DataLoader(train_dataset, batch_size=train_nsamples, shuffle=True)
test_dataloader = DataLoader(train_dataset, batch_size=test_nsamples, shuffle=True)