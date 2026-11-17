import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from dados import preparar_dados


TAXA_APRENDIZADO = 0.001
EPOCAS = 12
BATCH_SIZE = 256


class RegistrarErros(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.erros = []

    def on_train_batch_end(self, batch, logs=None):
        self.erros.append(float(logs["loss"]))


def criar_modelo(quantidade_atributos, ativacao):
    modelo = keras.Sequential([
        keras.layers.Input(shape=(quantidade_atributos,)),
        keras.layers.Dense(32, activation=ativacao),
        keras.layers.Dense(16, activation=ativacao),
        keras.layers.Dense(8, activation=ativacao),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=TAXA_APRENDIZADO),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return modelo


def gerar_grafico(historicos, metrica, titulo, arquivo):
    plt.figure(figsize=(8, 5))

    for ativacao, historico in historicos.items():
        plt.plot(historico[metrica], label=f"{ativacao} - treino")
        plt.plot(historico[f"val_{metrica}"], label=f"{ativacao} - validação")

    plt.title(titulo)
    plt.xlabel("Época")
    plt.ylabel(titulo)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(arquivo)
    plt.close()


def main():
    X_treino, X_teste, y_treino, y_teste = preparar_dados()

    X_treino = X_treino.astype("float32")
    X_teste = X_teste.astype("float32")

    historicos = {}
    resultados = []

    for ativacao in ["relu", "tanh"]:
        print(f"\nTreinando MLP com ativação {ativacao.upper()}")

        keras.utils.set_random_seed(42)
        modelo = criar_modelo(X_treino.shape[1], ativacao)
        registro = RegistrarErros()

        historico = modelo.fit(
            X_treino, y_treino,
            validation_split=0.2,
            epochs=EPOCAS,
            batch_size=BATCH_SIZE,
            callbacks=[registro],
            verbose=1
        )

        historicos[ativacao] = historico.history

        print("Vetor de erros por época:",
              np.round(historico.history["loss"], 4))

        pd.DataFrame({
            "iteracao": range(1, len(registro.erros) + 1),
            "erro": registro.erros
        }).to_csv(f"erros_{ativacao}.csv", index=False)

        perda, acuracia = modelo.evaluate(
            X_teste, y_teste, verbose=0
        )

        probabilidades = modelo.predict(
            X_teste, verbose=0
        ).ravel()

        previsoes = (probabilidades >= 0.5).astype(int)

        resultados.append({
            "ativacao": ativacao,
            "perda": round(perda, 4),
            "acuracia": round(acuracia, 4),
            "precisao": round(precision_score(
                y_teste, previsoes, zero_division=0
            ), 4),
            "recall": round(recall_score(
                y_teste, previsoes, zero_division=0
            ), 4),
            "f1": round(f1_score(
                y_teste, previsoes, zero_division=0
            ), 4)
        })

    tabela = pd.DataFrame(resultados)

    print("\nComparação dos modelos:")
    print(tabela.to_string(index=False))

    tabela.to_csv("resultados.csv", index=False)

    gerar_grafico(
        historicos, "loss", "Perda", "grafico_perda.png"
    )

    gerar_grafico(
        historicos, "accuracy", "Acurácia", "grafico_acuracia.png"
    )

    print("\nTreinamento e gráficos finalizados!")


if __name__ == "__main__":
    main()
