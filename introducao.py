# Rede neural com um perceptron de representação de uma reta
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

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

# Preparando a infraestrutura de dados
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

# Criação da rede
# Analisa se o vai usar a cpu ou gpu
device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f"Rodando na {device}")

# Lança o device pro modelo     
model = LineNetwork().to(device)

# Função de perda (loss) pelo erro quadrático médio
lossfunc = nn.MSELoss() # Mean Square Error Loss

# Gradiente Descendente Estocástico
# SGD = Stochastic Gradient Descent
# Taxa de aprendizado lr = learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
# função de treinamento e teste
def train(model, dataloader, lossfunc, optimizer):
    model.train()
    cumloss = 0.0
    # X - dado, y - resposta
    for X, y in dataloader:
        # Lançar o dado para o device(gpu ou cpu)
        # Unsqueeze - converte o dado assim: [[1], [2], [3]]
        X = X.unsqueeze(1).float().to(device) 
        y = y.unsqueeze(1).float().to(device)

        # Predição do modelo - o que ele acha
        pred = model(X)
        # Calculo da função de perda, ou seja, o que eu prediz x a verdade
        loss = lossfunc(pred, y)

        # Zera os gradientes, porque o pythorch acumula os gradientes
        optimizer.zero_grad()
        # Computa os gradientes - backpropagation
        loss.backward()
        # Anda, de fato, na direção que reduz o erro
        optimizer.step()
        # Loss é um tensor; item pra obter o float
        cumloss += loss.item()

        return cumloss / len(dataloader)
    
def test(model, dataloader, lossfunc):
    # Forma de avaliação so pra rodar o modelo
    model.eval()
    cumloss = 0.0
    with torch.no_grad(): # Não acumulo de gradientes
    # X - dado, y - resposta
        for X, y in dataloader:
            # Lançar o dado para o device(gpu ou cpu)
            # Unsqueeze - converte o dado assim: [[1], [2], [3]]
            X = X.unsqueeze(1).float().to(device) 
            y = y.unsqueeze(1).float().to(device)

            # Predição do modelo - o que ele acha
            pred = model(X)
            # Calculo da função de perda, ou seja, o que eu prediz x a verdade
            loss = lossfunc(pred, y)

            # Loss é um tensor; item pra obter o float
            cumloss += loss.item()

            return cumloss / len(dataloader)

# Pra  visualizar:
def plot_comparinson( f, model, interval=(-10, 10), nsamples=10 ):
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.grid(True, which= 'both')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')

    samples = np.linspace(interval[0], interval[1], nsamples)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(samples).unsqueeze(1).float().to(device))
    
    ax.plot(samples, list(map(f, samples)), 'o', label='ground truth')
    ax.plot(samples, pred.cpu(), label='model')
    plt.legend()
    plt.show()

# Treinando a rede, de fato
# Treinar 101 vezes
epochs = 201
for t in range(epochs):
    train_loss = train(model, train_dataloader, lossfunc, optimizer)
    if t % 40 == 0:
        print(f"Epoch: {t}; Train Loss: {train_loss}")
        plot_comparinson(line, model)

test_loss = test(model, test_dataloader, lossfunc)
print(f"Test Loss: {test_loss}")


#-------------------------------------------------
# Outro exemplo com uma função nao linear
#-------------------------------------------------

class MultiLayerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # 4 Camadas ocultas
            # Parte linear
                nn.Linear(1, 128),
                # Parte não linear
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.layers(x)
    
multimodel = MultiLayerNetwork().to(device)

# Função referência
from math import cos
f = lambda x:cos(x/2)
train_dataset = AlgebraicDataset(f, interval, train_nsamples)
test_dataset = AlgebraicDataset(f, interval, test_nsamples)

train_dataloader = DataLoader(train_dataset, train_nsamples, shuffle=True)
test_dataloader = DataLoader(test_dataset, test_nsamples, shuffle=True)

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f"Rodando na {device}")

lossfunc = nn.MSELoss()
optimizer = torch.optim.SGD(multimodel.parameters(), lr=1e-3)

epochs = 20001
for t in range(epochs):
    train_loss = train(multimodel, train_dataloader, lossfunc, optimizer)
    if t % 2000 == 0:
        print(f"Epoch: {t}; Train Loss: {train_loss}")
        plot_comparinson(f, multimodel, nsamples=40)

test_loss = test(multimodel, test_dataloader, lossfunc)
print(f"Test Loss: {test_loss}")

