## Introdução
1. Exploração de dados: A base escolhida para testar a decision tree foi a ["Red Wine Quality"](https://www.kaggle.com/datasets/lovishbansal123/red-wine-quality/data){:target='_blank'}, uma base limpa que dispõe de 11 variáveis que afetam a qualidade de um Vinho tinto, são elas: Fixed Acidity, Volatile Acidity, Citric Acid, Residual Sugar, Chlorides, Free Sulfur Dioxide, Total Sulfur Dioxide, Density, pH, Sulphates, Alcohol. Todas já estão de forma adequada para o modelo.(A coluna "Citric Acid" possui alguns valores 0.0, porém ainda é coerente, alguns vinhos não possuem acido citrico).  
A partir destas variáveis é possível prever a qualidade do vinho, por isso para o treinamento temos a coluna 'Quality' que contém uma nota de 0 a 10 para o vinho, sendo esta nossa variavel target.
    

## Divisão dos Dados

O conjunto de dados foi dividido em 80% para treino e 20% para um primeiro teste de validação


## Treinamento e Resultados


=== "Results"

    ```python exec="on" html="1"
    --8<-- "docs/KNN/knnteste.py"
    ```

=== "Code"

    ```python
    --8<-- "docs/KNN/knnteste.py"
    ```


## Avaliação do modelo

O modelo atingiu uma acurácia de 0,59, o que indica a necessidade de ajustes para melhorar sua precisão. Entre as possíveis estratégias estão:

Padronizar as variáveis (StandardScaler) e reduzir multicolinearidade (correlação/VIF).

Ajustar hiperparâmetros do KNN (k, weights, métrica) com validação cruzada.

Testar divisões de treino/teste (ex.: 70/30).

Avaliar com métricas mais informativas em caso de desbalanceamento: balanced accuracy e F1 macro.

Revisar as faixas de qualidade; se fizer sentido, testar KNN Regressor (nota 0–10) e discretizar depois