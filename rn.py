import numpy as np

# Dados de entrada (exemplo: [horas estudadas, exercícios feitos])

X = np.array([[1, 2], 
              [2, 3], 
              [3, 4], 
              [4, 5]])

# Saída esperada (exemplo: notas nas provas)
y = np.array([[5], [7], [9], [11]])

# Inicializa pesos e bias aleatórios
pesos = np.random.rand(2, 1)
bias = np.random.rand(1)

# Taxa de aprendizado
learning_rate = 0.01

# Treinamento da rede
for epoch in range(1000):  
    # Forward Pass (Multiplicação de Matriz)
    y_pred = np.dot(X, pesos) + bias

    # Cálculo do erro
    erro = y - y_pred

    # Ajuste dos pesos (Aprendizado) - Correção: Média para evitar atualização instável
    pesos += learning_rate * np.dot(X.T, erro) / len(X)
    bias += learning_rate * np.sum(erro) / len(X)

# Testando com um novo dado
novo_dado = np.array([[2, 3]])  
predicao = np.dot(novo_dado, pesos) + bias
print(f"Previsão para {novo_dado}: {predicao[0][0]:.2f}")
