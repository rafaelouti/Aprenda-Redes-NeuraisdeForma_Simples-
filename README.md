📢 Aprenda Redes Neurais de Forma Simples!
Você já ouviu falar de redes neurais mas achou complicado entender? Vou te mostrar de um jeito fácil e ainda compartilhar um código funcional! 🚀

🤔 O que é uma Rede Neural?
Pensa no seu cérebro: você recebe informações, processa e toma decisões. Uma rede neural faz o mesmo, mas usando matemática e aprendizado de máquina.

💡 Exemplo prático:
Imagine que queremos prever a nota de um aluno com base no tempo de estudo e nos exercícios feitos. Podemos ensinar uma rede neural a aprender essa relação!

🧑‍💻 Código Simples de Rede Neural em Python
Aqui está um modelo básico que aprende a prever uma nota com base em dados de entrada:

python
Copiar
Editar
import numpy as np

# Dados de entrada (horas estudadas, exercícios feitos)
X = np.array([[1, 2], 
              [2, 3], 
              [3, 4], 
              [4, 5]])

# Notas esperadas
y = np.array([[5], [7], [9], [11]])

# Inicializa pesos e bias aleatórios
pesos = np.random.rand(2, 1)
bias = np.random.rand(1)

# Taxa de aprendizado
learning_rate = 0.01

# Treinamento da rede (Ajuste dos pesos)
for epoch in range(1000):  
    y_pred = np.dot(X, pesos) + bias
    erro = y - y_pred
    pesos += learning_rate * np.dot(X.T, erro) / len(X)
    bias += learning_rate * np.sum(erro) / len(X)

# Testando com um novo aluno
novo_dado = np.array([[5, 6]])
predicao = np.dot(novo_dado, pesos) + bias
print(f"Previsão para {novo_dado}: {predicao[0][0]:.2f}")
🔍 O que esse código faz?
✅ Cria uma rede neural simples do zero
✅ Treina a rede para aprender a relação entre estudo e notas
✅ Faz previsões para novos alunos

📌 Esse é um exemplo básico, mas redes neurais mais complexas são usadas para reconhecimento facial, carros autônomos e muito mais!
