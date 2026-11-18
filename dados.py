from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ARQUIVO = Path(__file__).resolve().parent / "heart_disease_health_indicators_BRFSS2015.csv.zip"
ALVO = "HeartDiseaseorAttack"

# Dataset: Heart Disease Health Indicators (BRFSS 2015 - CDC / Kaggle).
# Contém 253.680 registros e 21 atributos clínicos, comportamentais e
# socioeconômicos (sem valores ausentes). Todos os 21 atributos são mantidos
# por representarem fatores de risco cardiovascular reconhecidos (pressão alta,
# colesterol, IMC, tabagismo, AVC, diabetes, atividade física, idade, etc.).



def preparar_dados():
    # Lê o CSV diretamente do arquivo ZIP.
    df = pd.read_csv(ARQUIVO)

    print("Quantidade de registros:", len(df))
    print("Quantidade de atributos:", len(df.columns) - 1)
    print("Valores ausentes:", df.isna().sum().sum())
    print("Distribuição da variável-alvo:")
    print(df[ALVO].value_counts().sort_index())

    # X contém os atributos; y contém a resposta que queremos prever.
    X = df.drop(columns=[ALVO])
    y = df[ALVO]

    # Separa 80% para treinamento e 20% para teste.
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Padroniza usando somente os dados de treinamento como referência.
    padronizador = StandardScaler()
    X_treino = padronizador.fit_transform(X_treino)
    X_teste = padronizador.transform(X_teste)

    return X_treino, X_teste, y_treino.to_numpy(), y_teste.to_numpy()


if __name__ == "__main__":
    X_treino, X_teste, y_treino, y_teste = preparar_dados()
    print("Treinamento:", X_treino.shape)
    print("Teste:", X_teste.shape)
