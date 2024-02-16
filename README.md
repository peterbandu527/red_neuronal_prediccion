# Programación Emergente PRE803

# Equipo
- Barrios Pedro CI:26590035
- Mardones Esperanza CI: 27793192
- Harley Sambrano CI: 26251830

# Documentación de la Red Neuronal LSTM para Predicción de Ventas

## Introducción

Este documento proporciona una descripción detallada de una red neuronal LSTM (Long Short-Term Memory) utilizada para predecir ventas. La red neuronal se entrena con datos históricos de ventas y realiza predicciones futuras.

## Estructura del Código

El código proporcionado consta de las siguientes secciones:

1. **Importación de bibliotecas:** Se importan las bibliotecas necesarias para el procesamiento de datos y la construcción del modelo.

2. **Generación de datos de ventas:** Se generan datos de ventas aleatorios para simular un conjunto de datos históricos.

3. **Normalización de datos:** Los datos de ventas se normalizan utilizando el escalador MinMaxScaler de scikit-learn.

4. **Creación de conjuntos de datos de entrenamiento y prueba:** Se crean conjuntos de datos de entrenamiento y prueba dividiendo los datos normalizados en secuencias de tiempo.

5. **Construcción del modelo LSTM:** Se define y compila el modelo LSTM utilizando TensorFlow.

6. **Entrenamiento del modelo:** El modelo se entrena con los datos de entrenamiento utilizando el optimizador Adam y la función de pérdida de error cuadrático medio (MSE).

7. **Predicciones:** Se realizan predicciones utilizando el modelo entrenado y se visualizan las ventas reales vs. las predicciones.

## Detalles del Modelo

El modelo de red neuronal LSTM consta de las siguientes capas:

- Capa LSTM (Long Short-Term Memory): Una capa LSTM con 256 unidades de memoria y una función de activación lineal.
  
- Capa de Dropout: Una capa de dropout con una tasa de dropout del 20%, que ayuda a prevenir el sobreajuste.
  
- Segunda Capa LSTM: Otra capa LSTM con 128 unidades de memoria y una función de activación lineal.
  
- Segunda Capa de Dropout: Otra capa de dropout con una tasa de dropout del 20%.
  
- Capa Densa: Una capa densa con una unidad de salida que produce la predicción.

## Entrenamiento del Modelo

El modelo se entrena durante 100 épocas con un tamaño de lote de 32. Se utiliza una división del 10% de los datos de entrenamiento como conjunto de validación. Los datos de entrenamiento se barajan antes de cada época.

## Visualización de Resultados

Se visualizan las ventas reales vs. las predicciones realizadas por el modelo en un gráfico, que muestra la comparación de las ventas reales y las ventas predichas para el conjunto de datos de prueba.

## Ejecución del Código

Para ejecutar el código, sigue estos pasos:

1. Clona el repositorio que contiene el código de la red neuronal LSTM.

2. Abre una terminal o línea de comandos y navega hasta el directorio donde se encuentra el código.

3. Ejecuta el siguiente comando para instalar las dependencias necesarias:


4. Ejecuta el script Python que contiene el código de la red neuronal LSTM.

## Archivo de Requisitos

El archivo `requirements.txt` contiene una lista de las bibliotecas y sus versiones específicas necesarias para ejecutar el código. Puedes instalar estas dependencias ejecutando el comando `pip install -r requirements.txt` como se describe anteriormente.

## Conclusiones

El modelo LSTM proporciona una forma efectiva de predecir las ventas futuras basadas en datos históricos. Se puede ajustar el rendimiento del modelo mediante la modificación de la arquitectura de la red, los hiperparámetros de entrenamiento y los datos de entrada.
