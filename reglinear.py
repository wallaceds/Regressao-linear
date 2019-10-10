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


clientes = pd.read_csv("Ecommerce Customers")
clientes.head()
clientes.describe()
clientes.info ()


# ## Análise exploratória
# 

# In[ ]:


sns.set_style('whitegrid')
sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=clientes)
sns.jointplot(x="Time on App", y="Length of Membership", data=clientes, kind="hex")
sns.jointplot (x="Time on Website", y ="Yearly Amount Spent", data=clientes)
sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=clientes, kind="hex")
sns.pairplot (clientes)
sns.lmplot (x="Length of Membership", y="Yearly Amount Spent", data=clientes)


# ## Aplicando o modelo preditivo

# In[ ]:


y = clientes["Yearly Amount Spent"]
X = clientes[["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split (X,y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression ()

lm.fit(X_train, y_train)

prediction = lm.predict (X_test)

plt.figure(figsize=(10, 8), dpi=70)
plt.scatter (y_test, prediction, color ="#2E9AFE")
plt.title ("Visualizando os resultados")
plt.xlabel ("y de teste")
plt.ylabel ("y predito")
plt.show ()

from sklearn import metrics

print("MAE:", metrics.mean_absolute_error (y_test, prediction))
print("MSE:", metrics.mean_squared_error (y_test, prediction))
print("RMSE:", np.sqrt(metrics.mean_squared_error (y_test, prediction)))

sns.distplot ((y_test-prediction), bins=50)

print("Número de Coeficientes: ", len(lm.coef_))
lm.coef_

coef = pd.DataFrame(lm.coef_, X.columns, columns=['Coefs'])
coef


# ## Conclusão

# 1 - Mantendo as demais variáveis constantes, cada uma unidade de uma variável acima que for aumentada, irá aumentar o valor
# do coeficiente em dólares. Portanto, a empresa deve orientar-se em fidelizar os clientes, pois a cada uma unidade aumentada de 
# "Length of membership" (anos como cliente), a loja tem o aumento de 61 dólares. 
# 
# 2 - A empresa deve focar seus esforços no aplicativo que obtem relativamente um lucro maior do que no website, pois cada unidade de tempo no aplicativo está associado a 38 dólares de lucro enquanto o website apenas 0,19 centavos de dólar. 
