#!/usr/bin/env python
# coding: utf-8

# # Regressão Linear múltipla

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Fazendo a leitura do arquivo
clientes = pd.read_csv("Ecommerce Customers")


# In[ ]:


#verificando o arquivo
clientes.head()


# In[ ]:


#Verificando as medidas descritivas
clientes.describe()


# In[ ]:


#Obtendo informações dos dados. O dataset obtém 500 linhas e 8 colunas.
clientes.info ()


# ## Análise exploratória
# 

# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


# Aqui é feito um comparativo do tempo do cliente no APP e do valor gasto anualmente
sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=clientes)


# In[ ]:


# # Aqui é feito um comparativo do tempo do cliente no APP e dos anos que o cliente é membro da loja.
sns.jointplot(x="Time on App", y="Length of Membership", data=clientes, kind="hex")


# In[ ]:


# Aqui é feito um comparativo do tempo do cliente no Website e do valor gasto anualmente
sns.jointplot (x="Time on Website", y ="Yearly Amount Spent", data=clientes)


# In[ ]:


# Aqui é feito um comparativo do tempo do cliente no Website e dos anos que o cliente é membro da loja.
sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=clientes, kind="hex")


# In[ ]:


# Aqui é utilizado o recurso do Pairplot para avaliar a relação entre o conjunto de dados.
sns.pairplot (clientes)


# In[ ]:


# Fica evidente que a melhor correlação do valor gasto anualmente "Yearly amount spent" foi com o do período 
# como membro "Length of Membership". Isso pode ser visto acima, e será detalhado abaixo:


# In[ ]:


sns.lmplot (x="Length of Membership", y="Yearly Amount Spent", data=clientes)


# ## Aplicando o modelo preditivo

# In[ ]:


# definindo minha variável preditiva (output)
y = clientes["Yearly Amount Spent"]


# In[ ]:


# definindo minhas variáveis preditoras (imput)
X = clientes[["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]]


# In[ ]:


# importando, definindo e dividindo os dados para treino e teste.
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split (X,y, test_size=0.3, random_state=101)


# In[ ]:


# Importando o linear regression
from sklearn.linear_model import LinearRegression


# In[ ]:


# Criando a instância
lm = LinearRegression ()


# In[ ]:


# Treinando o modelo
lm.fit(X_train, y_train)


# In[ ]:


# fazendo a predição dos dados de teste
prediction = lm.predict (X_test)


# In[ ]:


# criando um gráfico de dispersão para um comparativo entre o y de teste e o predito.
plt.figure(figsize=(10, 8), dpi=70)
plt.scatter (y_test, prediction, color ="#2E9AFE")
plt.title ("Visualizando os resultados")
plt.xlabel ("y de teste")
plt.ylabel ("y predito")
plt.show ()


# In[ ]:


# calculando o erro e avaliando o desempenho do modelo 
from sklearn import metrics

print("MAE:", metrics.mean_absolute_error (y_test, prediction))
print("MSE:", metrics.mean_squared_error (y_test, prediction))
print("RMSE:", np.sqrt(metrics.mean_squared_error (y_test, prediction)))


# In[ ]:


sns.distplot ((y_test-prediction), bins=50)


# In[ ]:


# Coeficientes
print("Número de Coeficientes: ", len(lm.coef_))
lm.coef_


# In[ ]:


# Coeficientes
coef = pd.DataFrame(lm.coef_, X.columns, columns=['Coefs'])
coef


# ## Conclusão

# 1 - Mantendo as demais variáveis constantes, cada uma unidade de uma variável acima que for aumentada, irá aumentar o valor
# do coeficiente em dólares. Portanto, a empresa deve orientar-se em fidelizar os clientes, pois a cada uma unidade aumentada de 
# "Length of membership" (anos como cliente), a loja tem o aumento de 61 dólares. 
# 
# 2 - A empresa deve focar seus esforços no aplicativo que obtem relativamente um lucro maior do que no website, pois cada unidade de tempo no aplicativo está associado a 38 dólares de lucro enquanto o website apenas 0,19 centavos de dólar. 
