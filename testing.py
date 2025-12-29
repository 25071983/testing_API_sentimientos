
import pandas as pd
import random
#import openpyxl


PATH_ARCHIVO_PRUEBA="./prueba01.xlsx"

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
    return random.choice([-1, 0, 1])

# Recorrer cada hoja
for nombre_hoja, df in hojas.items():
    # Asumimos:
    # columna 0 -> id
    # columna 1 -> comentario
    # columna 2 -> sentimiento

    # Procesar comentarios
    df["inferencia"] = df.iloc[:, 1].apply(procesar_comentario)

    hojas_procesadas[nombre_hoja] = df

# Guardar nuevo Excel
salida = "./output/comentarios_procesados.xlsx"
with pd.ExcelWriter(salida, engine="openpyxl") as writer:
    for nombre_hoja, df in hojas_procesadas.items():
        df.to_excel(writer, sheet_name=nombre_hoja, index=False)

print("Archivo procesado generado:", salida) 