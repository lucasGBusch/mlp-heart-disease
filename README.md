# MLP - Indicadores de Doenças Cardíacas

Projeto da **Atividade 02 de Sistemas Inteligentes** — Aplicação de aprendizado supervisionado e Redes Neurais Multicamadas (MLP) em dados reais de saúde pública.

## Objetivo

Classificar a presença ou histórico de doença cardíaca/infarto (`HeartDiseaseorAttack`) a partir de 21 indicadores clínicos, comportamentais e socioeconômicos, comparando o desempenho de uma rede MLP com diferentes funções de ativação (`ReLU` e `Tanh`).

## Dataset

- **Nome:** Heart Disease Health Indicators Dataset (BRFSS 2015)
- **Fonte:** Behavioral Risk Factor Surveillance System (BRFSS) — Centers for Disease Control and Prevention (CDC), disponibilizado no Kaggle
- **Período:** 2015
- **Arquivo:** `heart_disease_health_indicators_BRFSS2015.csv.zip`
- **Quantidade de amostras:** 253.680 registros (0 valores ausentes)
- **Atributos de entrada (21):** `HighBP`, `HighChol`, `CholCheck`, `BMI`, `Smoker`, `Stroke`, `Diabetes`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `GenHlth`, `MentHlth`, `PhysHlth`, `DiffWalk`, `Sex`, `Age`, `Education`, `Income`
- **Variável-alvo (target):** `HeartDiseaseorAttack` (`0.0`: 229.787 registros / ~90,58%; `1.0`: 23.893 registros / ~9,42%)

### Justificativa dos atributos e pré-processamento

Todos os 21 atributos disponíveis foram mantidos por representarem fatores de risco cardiovascular amplamente documentados na literatura médica (hipertensão, colesterol alto, IMC, tabagismo, AVC, diabetes, sedentarismo, hábitos alimentares, saúde geral, faixa etária e condições socioeconômicas).

Em `dados.py`:
- Os dados são divididos de forma estratificada em **80% treino** (202.944 amostras) e **20% teste** (50.736 amostras), com `random_state=42`.
- A padronização é feita com `StandardScaler` ajustado exclusivamente sobre o conjunto de treino (`fit_transform`) e aplicado ao conjunto de teste (`transform`), evitando vazamento de dados (*data leakage*).

## Arquitetura da MLP e Hiperparâmetros

Em `modelo.py`, a rede neural implementada em Python com Keras/TensorFlow possui:
- **Camada de entrada:** 21 atributos
- **3 camadas ocultas densas:** `Dense(32)` → `Dense(16)` → `Dense(8)`
- **Camada de saída:** `Dense(1, activation="sigmoid")` (classificação binária)
- **Funções de ativação testadas:** `relu` e `tanh`
- **Otimizador e perda:** `Adam(learning_rate=0.001)` e `binary_crossentropy`

### Justificativa dos parâmetros

- **Taxa de aprendizado (`0.001`):** valor padrão do otimizador Adam, oferecendo equilíbrio entre estabilidade na atualização dos pesos e velocidade de convergência.
- **Épocas (`12`):** quantidade suficiente para estabilização das curvas de perda e acurácia sem sobreajuste prolongado.
- **Tamanho do batch (`256`):** adequado ao grande volume de dados (~162.355 amostras efetivas de treino após `validation_split=0.2`), garantindo gradientes estáveis e boa eficiência computacional.

## Como Executar

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. (Opcional) Verifique o carregamento e pré-processamento dos dados:

```bash
python dados.py
```

3. Execute o treinamento, avaliação e geração dos gráficos:

```bash
python modelo.py
```

## Saídas Geradas

Ao executar `modelo.py`, são exibidos no terminal os vetores de erro por época e por iteração (*batch*), além da tabela comparativa no conjunto de teste. Também são gerados:
- `resultados.csv`: perda, acurácia, precisão, recall e F1-score para `relu` e `tanh`
- `erros_relu.csv` e `erros_tanh.csv`: histórico de erro (loss) por iteração
- `grafico_perda.png` e `grafico_acuracia.png`: curvas de treino e validação
