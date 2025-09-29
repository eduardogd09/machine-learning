import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.figure(figsize=(12, 10))

df = pd.read_csv('https://raw.githubusercontent.com/eduardogd09/machine-learning/refs/heads/main/winequality-red.csv')
# Carregar o conjunto de dados
x = df[['fixed acidity', 'volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]


q = df['quality']

# 4 categorias sem sobreposição: 8 vai para a faixa 8–10
condicoes = [
    q < 3,                         # Cat 0
    (q >= 3) & (q <= 5),           # Cat 1
    (q >= 6) & (q < 8),            # Cat 2  (OBS: < 8)
    (q >= 8) & (q <= 10)           # Cat 3  (OBS: >= 8)
]
labels = [0, 1, 2, 3]
cat_arr = np.select(condicoes, labels, default=np.nan)
df['quality_cat'] = pd.Series(cat_arr, index=df.index).astype('Int64')


# (opcional) rótulos textuais
label_txt = {0: '<3', 1: '3–5', 2: '6–8', 3: '8–10'}
df['quality_cat_txt'] = df['quality_cat'].map(label_txt)

y = df['quality_cat']

# Dividir os dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# Reduzir para 2 dimensões (apenas para visualização)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Treinar KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)


# Visualize decision boundary
h = 0.02  # Step size in mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, style=y, palette="deep", s=100)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("KNN Decision Boundary (k=5)")

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())