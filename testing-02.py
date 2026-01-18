
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from matplotlib.colors import LinearSegmentedColormap

from sentiment_model import SentimentModel

MODEL_PATH = "../train/output/modelo_final.pkl"
VECT_PATH = "../train/output/vectorizador_final.pkl"

model = SentimentModel(MODEL_PATH, VECT_PATH)

#import openpyxl


PATH_ARCHIVO_PRUEBA="./prueba02.xlsx"

print (f"leyendo xls '{PATH_ARCHIVO_PRUEBA}'")
hojas = pd.read_excel(PATH_ARCHIVO_PRUEBA, sheet_name=None)

#print(df.head())

#df = pd.read_excel("archivo.xls", sheet_name="Hoja1")

# Diccionario para guardar las hojas procesadas
hojas_procesadas = {}

def procesar_comentario(texto):
    """
    Aquí va tu lógica real:
    análisis, llamada a API, modelo, etc.
    """
    #return random.choice([-1, 0, 1])

    pred = model.predict(texto)
    return pred

print("Inicio procesamiento")
# Recorrer cada hoja
for nombre_hoja, df in hojas.items():
    # Asumimos:
    # columna 0 -> id
    # columna 1 -> comentario
    # columna 2 -> sentimiento
    if len(df) < 5:

        print(f"saltando hoja: {nombre_hoja} ")
    else:
        print(f"procesando hoja: {nombre_hoja} ")

        # Procesar comentarios y agrgar columna inferencia 
        df["inferencia"] = df.iloc[:, 1].apply(procesar_comentario)


        hojas_procesadas[nombre_hoja] = df

# Guardar nuevo Excel
salida = "./output/comentarios_procesados-02.xlsx"
with pd.ExcelWriter(salida, engine="openpyxl") as writer:
    for nombre_hoja, df in hojas_procesadas.items():
        df.to_excel(writer, sheet_name=nombre_hoja, index=False)

print("Archivo procesado generado:", salida) 

def accuracy_global(hojas_procesadas,
                    col_real="Codigo",
                    col_pred="inferencia"):
    """
    Calcula el accuracy global considerando todas las hojas.

    :param hojas_procesadas: dict {nombre_hoja: DataFrame}
    :param col_real: nombre de la columna con el valor real
    :param col_pred: nombre de la columna con el valor predicho
    :return: accuracy global (float)
    """
    total_correctos = 0
    total_registros = 0

    for df in hojas_procesadas.values():
        total_correctos += (df[col_real] == df[col_pred]).sum()
        total_registros += len(df)

    if total_registros == 0:
        return 0.0

    return total_correctos / total_registros


def clasificar_accuracy(acc):
    if acc < 0.5:
        return "Modelo muy débil"
    elif acc < 0.65:
        return "Modelo débil / poco confiable"
    elif acc < 0.75:
        return "Desempeño aceptable"
    elif acc < 0.85:
        return "Buen desempeño"
    else:
        return "Muy buen / excelente desempeño"

acc = accuracy_global(hojas_procesadas)
print(f"Accuracy global: {acc:.4f}")
print(f"Accuracy : {clasificar_accuracy(acc)}")

def accuracy_por_hoja(hojas_procesadas,
                      col_real="Codigo",
                      col_pred="inferencia",
                      min_filas=5):
    """
    Calcula el accuracy por hoja, lo clasifica y devuelve un array con el resultado.

    :return: list[dict]
    """
    resultados = []

    for nombre_hoja, df in hojas_procesadas.items():
        if len(df) < min_filas:
            acc = 0.0
        else:
            correctos = (df[col_real] == df[col_pred]).sum()
            acc = correctos / len(df)

        evaluacion = clasificar_accuracy(acc)

        resultados.append({
            "hoja": nombre_hoja,
            "accuracy": acc,
            "evaluacion": evaluacion
        })

    return resultados


acc = accuracy_por_hoja(hojas_procesadas)
print("Accuracy por hoja")
print(acc)



def generar_matriz_confusion(
    hojas_procesadas,
    col_real="Codigo",
    col_pred="inferencia",
    etiquetas=(1, 0),
    nombres_etiquetas=("Positivo","Negativo"),
    archivo_salida="./output/matriz_confusion.jpg"
):
    """
    Genera la matriz de confusión global y la guarda como imagen JPG.

    Parámetros:
        hojas_procesadas (dict): {nombre_hoja: DataFrame}
        col_real (str): columna ground truth
        col_pred (str): columna predicción
        etiquetas (tuple): valores numéricos de las clases
        nombres_etiquetas (tuple): nombres legibles de las clases
        archivo_salida (str): nombre del archivo JPG
    """

    # Unificar todas las hojas
    df_total = pd.concat(hojas_procesadas.values(), ignore_index=True)

    # Calcular matriz de confusión
    cm = confusion_matrix(
        df_total[col_real],
        df_total[col_pred],
        labels=etiquetas
    )

    # Normalizar por fila para mejor interpretación
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    # Crear colormap personalizado:
    # Verde para diagonal, amarillo/naranjo para errores
    colors = ["#FFF3B0", "#F4A261", "#2A9D8F"]
    cmap = LinearSegmentedColormap.from_list("custom_cm", colors)

    # Crear figura
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_norm, cmap=cmap)
    plt.title("Matriz de Confusión Global")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")

    # Etiquetas de ejes
    plt.xticks(range(len(nombres_etiquetas)), nombres_etiquetas)
    plt.yticks(range(len(nombres_etiquetas)), nombres_etiquetas)

    # Mostrar valores dentro de la matriz
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(archivo_salida, dpi=300)
    plt.close()

print("\n\n generar matriz de confusion")

generar_matriz_confusion(
    hojas_procesadas,
    archivo_salida="./output/matriz_confusion_sentimientos-02.jpg"
)

def generar_matrices_confusion_por_hoja(
    hojas_procesadas,
    col_real="Codigo",
    col_pred="inferencia",
    etiquetas=(1, 0),
    nombres_etiquetas=("Positivo", "Negativo"),
    carpeta_salida="./output"
):
    """
    Genera una matriz de confusión por cada hoja y la guarda como imagen JPG.

    Parámetros:
        hojas_procesadas (dict): {nombre_hoja: DataFrame}
        col_real (str): columna ground truth
        col_pred (str): columna predicción
        etiquetas (tuple): valores numéricos de las clases
        nombres_etiquetas (tuple): nombres legibles de las clases
        carpeta_salida (str): carpeta donde se guardan las imágenes
    """

    for nombre_hoja, df in hojas_procesadas.items():

        # Saltar hojas vacías
        if df.empty:
            continue

        # Calcular matriz de confusión
        cm = confusion_matrix(
            df[col_real],
            df[col_pred],
            labels=etiquetas
        )

        # Normalizar por fila
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        # Colormap personalizado
        colors = ["#FFF3B0", "#F4A261", "#2A9D8F"]
        cmap = LinearSegmentedColormap.from_list("custom_cm", colors)

        # Crear figura
        plt.figure(figsize=(6, 5))
        plt.imshow(cm_norm, cmap=cmap)
        plt.title(f"Matriz de Confusión - {nombre_hoja}")
        plt.xlabel("Predicción")
        plt.ylabel("Valor real")

        # Etiquetas
        plt.xticks(range(len(nombres_etiquetas)), nombres_etiquetas)
        plt.yticks(range(len(nombres_etiquetas)), nombres_etiquetas)

        # Valores en la matriz (valores absolutos)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")

        # Nombre del archivo
        archivo_salida = f"{carpeta_salida}/matriz_confusion-{nombre_hoja}.jpg"

        plt.tight_layout()
        plt.savefig(archivo_salida, dpi=300)
        plt.close()

generar_matrices_confusion_por_hoja(hojas_procesadas)

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def calcular_metricas_por_clase(
    hojas_procesadas,
    col_real="Codigo",
    col_pred="inferencia",
    etiquetas=(1, 0),
    nombres_etiquetas=("Positivo","Negativo"),
):
    """
    Calcula Precision, Recall y F1-score por clase.

    Retorna:
        DataFrame con métricas por clase
    """

    # Unificar todas las hojas
    df_total = pd.concat(hojas_procesadas.values(), ignore_index=True)

    precision, recall, f1, support = precision_recall_fscore_support(
        df_total[col_real],
        df_total[col_pred],
        labels=etiquetas,
        zero_division=0
    )

    # Construir DataFrame de salida
    metricas_df = pd.DataFrame({
        "Clase": nombres_etiquetas,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Soporte": support
    })

    return metricas_df

print("\n\n calcular metrica por clases")

metricas = calcular_metricas_por_clase(hojas_procesadas)
print(metricas)

def analizar_metricas_combinadas(
    precision,
    recall,
    f1,
    nombre_clase=""
):
    etiqueta = f"Clase {nombre_clase}" if nombre_clase else "Clase"

    # Nivel general basado en F1
    if f1 < 0.50:
        nivel = "Muy deficiente"
    elif f1 >= 0.50 and f1 < 0.65:
        nivel = "Débil"
    elif f1 >= 0.65 and f1 < 0.75:
        nivel = "Aceptable"
    elif f1 >= 0.75 and f1 < 0.85:
        nivel = "Bueno"
    else:
        nivel = "Muy bueno / excelente"

    # Análisis del balance Precision–Recall
    if precision >= 0.75 and recall >= 0.75:
        balance = "Buen equilibrio entre detección y exactitud"
    elif precision >= 0.75 and recall < 0.65:
        balance = "Modelo conservador (pocos falsos positivos, pierde casos)"
    elif precision < 0.65 and recall >= 0.75:
        balance = "Modelo arriesgado (detecta muchos, pero se equivoca)"
    else:
        balance = "Balance intermedio o inestable"

    print(f"{etiqueta}: {nivel}")
    print(f"  • F1-score: {f1:.2f}")
    print(f"  • Precision: {precision:.2f}, Recall: {recall:.2f}")
    print(f"  • Interpretación: {balance}")

for _, fila in metricas.iterrows():
    analizar_metricas_combinadas(
        precision=fila["Precision"],
        recall=fila["Recall"],
        f1=fila["F1-score"],
        nombre_clase=fila["Clase"]
    )
    print("-" * 60)

