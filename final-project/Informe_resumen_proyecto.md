# Informe sobre el Entrenamiento y Evaluación de un Modelo de Clasificación de Imágenes usando ViT (Visual Transformers)

### 1. Introducción

El objetivo principal de este proyecto es desarrollar un sistema capaz de clasificar imágenes en diferentes categorías con alta precisión. Para lograr esto, se emplea un enfoque basado en técnicas de aprendizaje profundo, específicamente utilizando el modelo Vision Transformer (ViT). El modelo ViT es una arquitectura de redes neuronales que ha demostrado ser altamente efectiva en tareas de visión por computadora al tratar la entrada de imágenes como una secuencia de tokens. Esta técnica ha mostrado resultados prometedores en comparación con las arquitecturas convolucionales tradicionales, especialmente cuando se enfrenta a conjuntos de datos grandes y complejos.

### 2. Preparación de Datos

Antes de entrenar un modelo de clasificación de imágenes, es crucial preparar los datos de manera adecuada. Esto incluye la carga de las imágenes desde un archivo comprimido, la aplicación de transformaciones para mejorar la calidad y la diversidad de los datos, y la división del conjunto de datos en conjuntos de entrenamiento y validación. La división del conjunto de datos permite evaluar el rendimiento del modelo en datos que no ha visto durante el entrenamiento, lo que proporciona una evaluación más realista de su capacidad para generalizar a nuevos datos.

### 3. Definición del Modelo

En este proyecto, se utiliza la arquitectura Vision Transformer (ViT) como modelo base para la clasificación de imágenes. El modelo ViT trata la entrada de imágenes como una secuencia de tokens, lo que le permite capturar relaciones de largo alcance entre los diferentes píxeles de la imagen. Esta arquitectura se elige por su capacidad para aprender representaciones de alto nivel de las imágenes y su eficacia en una variedad de tareas de visión por computadora.

Para adaptar el modelo ViT a nuestro problema específico de clasificación de imágenes, se reemplaza la capa de salida para que coincida con el número de clases en nuestro conjunto de datos. Además, se congela el resto de los parámetros del modelo preentrenado para evitar que se modifiquen durante el entrenamiento, lo que acelera el proceso y evita el sobreajuste.

### 4. Entrenamiento del Modelo

Durante el entrenamiento del modelo, se utiliza un optimizador y una función de pérdida para ajustar los parámetros del modelo con el fin de minimizar la pérdida en el conjunto de datos de entrenamiento. Se realiza un seguimiento del progreso del entrenamiento mediante métricas como la pérdida y se ajustan los hiperparámetros según sea necesario para mejorar el rendimiento del modelo. Además, se utiliza la aceleración de GPU cuando está disponible para acelerar el proceso de entrenamiento y manejar conjuntos de datos más grandes de una mejor forma ya que se estaría trabajando en paralelo.

### 5. Evaluación del Modelo

Después de completar el entrenamiento del modelo, se evalúa su rendimiento en un conjunto de datos de validación separado. Se calculan diversas métricas de evaluación, como la precisión, la precisión equilibrada y el coeficiente Kappa de Cohen, para evaluar la capacidad del modelo para realizar predicciones precisas en datos no vistos. Estas métricas proporcionan información sobre la capacidad del modelo para generalizar a nuevos datos y su capacidad para distinguir entre diferentes clases.

### 6. Visualización de Resultados

Para comprender mejor el rendimiento del modelo, se visualizan los resultados utilizando gráficos y representaciones visuales. Esto incluye gráficas de pérdida durante el entrenamiento para monitorear el progreso del modelo, matrices de confusión que muestran las predicciones del modelo frente a las etiquetas verdaderas, y visualizaciones individuales de muestras del conjunto de validación con sus etiquetas verdaderas y predicciones del modelo. Estas visualizaciones proporcionan información sobre los aciertos y errores del modelo, lo que ayuda a identificar áreas de mejora y a comprender su comportamiento en diferentes escenarios.

### 7. Resultados

- Rendimiento del Modelo: Los resultados del entrenamiento del modelo son bastante prometedores. Durante las 20 épocas de entrenamiento, se observa una disminución constante en la función de pérdida tanto en el conjunto de entrenamiento como en el de validación. Esto sugiere que el modelo está aprendiendo de manera efectiva y generalizando bien a datos no vistos. El coeficiente de Kappa de Cohen de validación también aumenta a lo largo del entrenamiento, indicando una mejor concordancia entre las predicciones y las etiquetas reales.

- Precisión y Métricas de Evaluación: Se obtienen resultados bastante sólidos. La precisión de validación, que mide la proporción de predicciones correctas sobre el total de predicciones, se mantiene alta, alcanzando un valor máximo de alrededor del 96.84%. Además, se observa que tanto la precisión equilibrada como el coeficiente Kappa de Cohen también alcanzan valores altos y estables, lo que indica que el modelo es capaz de clasificar de manera efectiva todas las clases de manera equitativa y consistente.

- Estabilidad y Generalización: Aunque hubo algunas advertencias sobre el número de procesos trabajadores creados por el DataLoader y la incompatibilidad potencial con el código multihilo, lo que puede conducir a problemas de bloqueo y afectar la estabilidad del modelo, el proceso de entrenamiento fue relativamente estable y no se encontraron problemas significativos. Además, el modelo demostró una capacidad de generalización satisfactoria, ya que logró una precisión alta y consistente en el conjunto de validación.

- Tiempo de Ejecución: El tiempo total de ejecución del código fue de aproximadamente 3793.4880 segundos (1 hora y 3 minutos aproximadamente). Aunque este tiempo puede variar según el hardware y otros factores, proporciona una indicación general del costo computacional asociado con el entrenamiento del modelo.

### 8. Conclusión

En conclusión, el entrenamiento y evaluación de modelos de clasificación de imágenes es un proceso complejo que requiere una cuidadosa preparación de los datos, selección y adaptación del modelo, entrenamiento efectivo y evaluación exhaustiva del rendimiento. Al seguir un enfoque sistemático y utilizar herramientas y técnicas adecuadas, es posible desarrollar modelos de clasificación de imágenes altamente efectivos que puedan aplicarse en una variedad de escenarios del mundo real. El uso del modelo Vision Transformer (ViT) en este proyecto demuestra su eficacia y versatilidad en tareas de visión por computadora, destacando su potencial para abordar problemas complejos en este campo en constante evolución.
