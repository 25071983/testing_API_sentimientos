# Motor de IA: ml-service (FastAPI)

Este servicio representa el núcleo inteligente del sistema, encargado de transformar lenguaje natural en predicciones de sentimiento mediante modelos de Machine Learning avanzados.

---

## Funciones del Servicio (v3.0.1)

1.  **Pipeline Unificado**: Carga un objeto serializado (`.pkl`) que integra tanto el Vectorizador (TF-IDF) como el Clasificador (Logistic Regression), garantizando consistencia en el preprocesamiento.
2.  **Clasificación Tri-modal**: A diferencia de versiones anteriores, el sistema ahora clasifica en tres categorías:
    *   **Positivo**: Probabilidad ≥ 0.6
    *   **Neutro**: 0.4 < Probabilidad < 0.6
    *   **Negativo**: Probabilidad ≤ 0.4
3.  **Inferencia de Alta Velocidad**: Optimizado para responder en milisegundos tanto en peticiones unitarias como en procesamiento por lotes (Batch).
4.  **Cálculo de Confianza Dinámico**: Proporciona el nivel exacto de seguridad estadística para cada predicción.

---

## Especificaciones Técnicas

*   **Runtime**: Python 3.12 + FastAPI.
*   **Modelo Base**: Regresión Logística entrenada sobre el dataset R5K.
*   **Serialización**: Joblib para carga eficiente de modelos pesados.
*   **Versionamiento**: v3.0.1 (Soporte nativo para categoría Neutro).

---

## Arquitectura de Endpoints

1.  `POST /predict`: Análisis unitario. Retorna sentimiento, probabilidad y versión del modelo.
2.  `POST /predict/batch`: Procesamiento masivo concurrente.
3.  `GET /health`: Diagnóstico de salud y estado de carga del modelo.
4.  `GET /docs`: Documentación interactiva OpenAPI (Swagger).

---

## Despliegue y Actualización

El motor permite actualizaciones transparentes:
*   El modelo debe ubicarse en la carpeta `/models`.
*   La lógica de umbrales (`LOW_THRESHOLD`, `HIGH_THRESHOLD`) es configurable directamente en el código para ajustar la sensibilidad de la categoría "Neutro".
