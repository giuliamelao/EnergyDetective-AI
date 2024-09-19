#!/usr/bin/env python
# coding: utf-8

# # Plano de Implementação de Modelo de Machine Learning
# 
# ### Este protótipo contém:
# ##### Geração de Dados Sintéticos baseados nas KPIs geradas pelo SmartGrid
# ##### Implementação de Modelo de Isolation Forest para detecção de anomalias
# ##### Identificação de padrões de consumo

#     

# #### Import de libraries e tools que serão utilizados

# In[1]:


import pandas as pd
import random
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np


# #### Dados Sintéticos 

# In[2]:


data = []
start_date = datetime(2023, 1, 1)
trend_coefficients = [random.uniform(0.95, 1.05) for _ in range(10)]

for home_id in range(10):
    current_date = start_date
    for _ in range(1000):
        timestamp = current_date.strftime("%Y-%m-%d %H:%M:%S")
        
        forward_active_energy = round(random.uniform(0.8, 1.2) * trend_coefficients[home_id], 2)
        reverse_active_energy = round(random.uniform(0.8, 1.2) * trend_coefficients[home_id], 2)
        
        home_data = {
            "Home_ID": home_id,
            "Timestamp": timestamp,
            "Forward_Active_Energy": forward_active_energy,
            "Reverse_Active_Energy": reverse_active_energy,
        }
        data.append(home_data)
        
        current_date += timedelta(minutes=5)

df = pd.DataFrame(data)
df


# ### Qualidade de Dados

# In[3]:


# Função lambda para identificação de dados duplicados

df2 = df.apply(lambda df: df.duplicated(), axis=1)
df2.sum()


# In[4]:


# Verificação de dados nulos

df.isna().sum()


# ### Boxplot
# É um método para demonstrar graficamente os grupos de variação de dados numéricos através de seus quartis. Os gráficos Boxplot são úteis para identificar dospersão de dados, simetria, outliers e posições.

# In[5]:


for c in df.columns:
    if df[c].dtype != 'object':  # Check if the column is numeric
        sns.boxplot(data=df, x=c)
        plt.title(f'Boxplot of {c}')
        plt.show()
        plt.close()


# #### Histogramas
# Este gráfico mostrará as distribuições de frequência, será possível identificar se cada característica é uma distribuição gaussiana ou não

# In[6]:


df2 = df[['Forward_Active_Energy', 'Reverse_Active_Energy']]

for c in df2:
    ax = sns.histplot(df, x=c)
    plt.show()
    plt.close()


#       

#        

# ### Isolation Forest

# In[7]:


selected_columns = ["Forward_Active_Energy", "Reverse_Active_Energy"]

X = df[selected_columns]

isolation_forest = IsolationForest(contamination=0.054)
isolation_forest.fit(X)
anomaly_scores = isolation_forest.predict(X)

df["Anomaly_Score"] = anomaly_scores

anomalies = df[df["Anomaly_Score"] == -1]
anomalies


# In[8]:


# Scatter Plot serve para verificar a distribuição de dados totais

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Forward_Active_Energy", y="Reverse_Active_Energy", marker="o", color="blue")
plt.xlabel("Forward_Active_Energy")
plt.ylabel("Reverse_Active_Energy")
plt.title("Scatter Plot")
plt.grid(True)
plt.show()


# In[9]:


# Este Scatter Plot mostra a distribuição de anomalias

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Forward_Active_Energy', y='Reverse_Active_Energy', data=anomalies, color='red')
plt.xlabel('Forward_Active_Energy')
plt.ylabel('Reverse_Active_Energy')
plt.title('Scatter Plot of Outliers')
plt.show()


# In[10]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=anomalies[['Forward_Active_Energy', 'Reverse_Active_Energy']], orient='h', palette='Set1')
plt.xlabel('Energy Values')
plt.title('Box Plot of Outliers')
plt.show()


# #### Este histograma faz a verificação entre o comportamento de consumo normal e comportamento anômalo

# In[11]:


plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='Forward_Active_Energy', bins=30, kde=True)
sns.histplot(data=anomalies, x='Forward_Active_Energy', bins=30, kde=True, color='red', label='Anomalies')
plt.xlabel('Forward_Active_Energy')
plt.ylabel('Count')
plt.title('Histogram of Forward_Active_Energy with Anomalies Highlighted')
plt.legend()
plt.show()


#      

#     

#     

# ## Clustering e Identificação de Padrões de Consumo

# In[12]:


X = df[["Forward_Active_Energy", "Reverse_Active_Energy"]]
k = 10
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)

df["Cluster_Label"] = kmeans.labels_

custom_cmap = plt.cm.get_cmap('tab10', k)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(df["Forward_Active_Energy"], df["Reverse_Active_Energy"], c=df["Cluster_Label"], cmap=custom_cmap)
plt.xlabel("Forward_Active_Energy")
plt.ylabel("Reverse_Active_Energy")
plt.title("K-Means Clustering")
plt.colorbar(scatter, label="Cluster")
plt.show()


#     

# ### Modelo de Train-Test

# In[13]:


X = df[['Forward_Active_Energy']]
y = df['Reverse_Active_Energy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[14]:


model = IsolationForest()
model.fit(X_train, y_train)


# In[15]:


preds = model.predict(X_test)


# ## Métricas de avaliação de Modelo Proposto

# In[16]:


mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mape = mean_absolute_percentage_error(y_test, preds)
print(f'MAE:  {mae}')
print(f'MSE:  {mse}')
print(f'RMSE: {rmse}')
print(f'MAPE: {mape}')


# In[17]:


y_train.mean()


# In[18]:


baseline = np.arange(len(y_test))
baseline.fill(y_train.mean())


# In[19]:


mae_baseline = mean_absolute_error(y_test, baseline)
mse_baseline = mean_squared_error(y_test, baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test, baseline))
mape_baseline = mean_absolute_percentage_error(y_test, baseline)


# In[20]:


print(f'MAE:  {mae_baseline}')
print(f'MSE:  {mse_baseline}')
print(f'RMSE: {rmse_baseline}')
print(f'MAPE: {mape_baseline}')


# In[21]:


print(f'MAE / MAE_BASELINE:   {mae_baseline/mae}')
print(f'MSE / MSE_BASELINE:   {mse_baseline/mse}')
print(f'RMSE / RMSE_BASELINE: {rmse_baseline/rmse}')
print(f'MAPE / MAPE_BASELINE: {mape_baseline/mape}')


# In[ ]:




