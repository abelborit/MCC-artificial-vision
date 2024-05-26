# Implementación de un Modelo de Aprendizaje Profundo para Clasificación de Imágenes utilizando el modelo ViT en un entorno de Google Colab

1. Descargar el dataset a utilizar desde https://data.mendeley.com/datasets/snkd93bnjr/1 y crear una carpeta en el drive llamada `Colab Notebooks` (la cual es la que se crea automáticamente al crear un nuveo proyecto de Notebook en Colab) y luego dentro crear una carpeta llamada `DatosProyecto` en la cual se colocorá el dataset descargado en .zip ya que más adelante se copiará ese dataset.zip en el proyecto de colab y se descomprimirá para utilizar sus datos

2. Realiza la instalación de dos paquetes de Python, fastbook y timm, utilizando el gestor de paquetes pip:

   ```python
   !pip install -Uqq fastbook timm
   ```

   - fastbook: está relacionado con el libro "Deep Learning for Coders with fastai & PyTorch" escrito por Jeremy Howard y Sylvain Gugger. Contiene utilidades y funciones útiles para trabajar con los conceptos y ejemplos presentados en el libro.

   - timm: es una biblioteca que proporciona implementaciones de modelos de aprendizaje profundo preentrenados y herramientas relacionadas con la visión por computadora. Ofrece una amplia variedad de arquitecturas de modelos, incluyendo algunas de las más nuevas y populares.

3. Importación de bibliotecas y módulos de Python las cuales proporcionan acceso a diversas funcionalidades y herramientas necesarias para cargar y procesar datos, definir y entrenar modelos de redes neuronales, evaluar el rendimiento del modelo y visualizar resultados en el código que sigue:

   ```python
   import os
   import re
   import zipfile
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from torch.utils.data import DataLoader, random_split
   from torchvision import transforms
   from torchvision.datasets import ImageFolder
   from torchvision import models
   import timm
   from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
   import matplotlib.pyplot as plt
   import numpy as np
   from PIL import Image
   from PIL import ImageOps
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
   import time
   ```

   - import os: Este módulo proporciona funciones para interactuar con el sistema operativo. Se utiliza para realizar operaciones relacionadas con archivos y directorios, como navegación de directorios, manipulación de rutas de archivos, etc.

   - import re: Este módulo proporciona operaciones de coincidencia de expresiones regulares. Se utiliza para buscar patrones en cadenas de texto y realizar manipulaciones basadas en esos patrones.

   - import zipfile: Este módulo proporciona clases para leer y escribir archivos ZIP. Se utiliza para trabajar con archivos comprimidos en formato ZIP, como extraer archivos de un archivo ZIP.

   - import torch: Este es el módulo principal de PyTorch, un framework de aprendizaje profundo. Proporciona estructuras de datos y funciones para crear y entrenar redes neuronales.

   - import torch.nn as nn: Este submódulo de PyTorch contiene las clases y funciones para definir y operar capas de redes neuronales. Se utiliza para construir modelos de redes neuronales con diferentes arquitecturas.

   - import torch.nn.functional as F: Este submódulo proporciona funciones de activación y funciones de pérdida comunes utilizadas en el entrenamiento de redes neuronales.

   - from torch.utils.data import DataLoader, random_split: Estas son clases de PyTorch para trabajar con conjuntos de datos y cargar datos en lotes durante el entrenamiento de redes neuronales. DataLoader se utiliza para cargar datos de un conjunto de datos, mientras que random_split se utiliza para dividir un conjunto de datos en conjuntos de entrenamiento y validación de manera aleatoria.

   - from torchvision import transforms: Este módulo proporciona clases y funciones para realizar transformaciones de datos en imágenes, como cambiar de tamaño, recortar, rotar, etc. Se utiliza comúnmente para preprocesar imágenes antes de alimentarlas a modelos de redes neuronales.

   - from torchvision.datasets import ImageFolder: Esta clase de torchvision se utiliza para crear un conjunto de datos de imágenes a partir de una estructura de directorios donde cada subdirectorio representa una clase de imagen.

   - from torchvision import models: Este módulo de torchvision proporciona implementaciones preentrenadas de modelos de redes neuronales para tareas de visión por computadora, como ResNet, VGG, etc.

   - import timm: Este es el paquete que proporciona implementaciones de modelos de aprendizaje profundo preentrenados y herramientas relacionadas con la visión por computadora.

   - from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay: Estas son clases y funciones proporcionadas por scikit-learn, una biblioteca de aprendizaje automático en Python. Se utilizan para calcular métricas de evaluación de modelos de clasificación, como la precisión balanceada, el coeficiente kappa de Cohen y la matriz de confusión.

   - import matplotlib.pyplot as plt: Este módulo se utiliza para crear gráficos y visualizaciones en Python.

   - import numpy as np: Numpy es una biblioteca fundamental para la computación científica en Python. Proporciona soporte para matrices y operaciones matemáticas en ellas.

   - from PIL import Image, ImageOps: El módulo PIL (Python Imaging Library) proporciona clases y funciones para abrir, manipular y guardar imágenes en diferentes formatos. Image se utiliza para operaciones básicas de imagen, mientras que ImageOps proporciona operaciones de procesamiento de imágenes como cambio de tamaño, rotación, etc.

   - import time: Este módulo proporciona funciones para medir el tiempo de ejecución del código. Se utiliza para medir el tiempo que lleva ejecutar ciertas partes del código.

4. Este fragmento de código realiza operaciones relacionadas con el manejo de archivos, directorios en Google Colab y Drive, es decir, se monta Google Drive en Colab, se define la ruta al directorio donde se encuentran los datos del proyecto en Google Drive y luego se copia y descomprime un archivo ZIP que contiene los datos del proyecto en el entorno de Colab. Finalmente, se imprime el tiempo que conlleva realizar esta operación:

   ```python
   # Montar Google Drive
   from google.colab import drive
   drive.mount('/content/gdrive')

   # Definir el directorio de los datos
   DIR = "/content/gdrive/MyDrive/Colab Notebooks/DatosProyecto"

   # Copiar y descomprir el archivo zip con los datos
   start_time = time.time()
   with zipfile.ZipFile(os.path.join(DIR, "PBC_dataset_normal_DIB.zip"), 'r') as zip_ref:
       zip_ref.extractall("/content")

   print("*"*80)
   print("Tiempo de ejecución para cargar y descomprimir los datos:", time.time() - start_time, "segundos")
   print("*"*80)
   ```

   - from google.colab import drive: Importa el módulo drive del paquete google.colab que proporciona funciones para interactuar con Google Drive desde un entorno de Colab.

   - drive.mount('/content/gdrive'): Monta Google Drive en la ubicación /content/gdrive. Esto permite acceder y manipular archivos almacenados en Google Drive desde el entorno de Colab.

   - DIR = "/content/gdrive/MyDrive/Colab Notebooks/DatosProyecto": Define la variable DIR como la ruta al directorio donde se encuentran los datos del proyecto en Google Drive. En este caso, parece que los datos del proyecto están almacenados en la carpeta DatosProyecto dentro de la carpeta Colab Notebooks en Google Drive.

   - start_time = time.time(): Registra el tiempo de inicio de la operación de carga y descompresión de datos.

   - with zipfile.ZipFile(os.path.join(DIR, "PBC_dataset_normal_DIB.zip"), 'r') as zip_ref:: Abre el archivo ZIP especificado (PBC_dataset_normal_DIB.zip) ubicado en el directorio definido anteriormente (DIR) en modo de solo lectura.

   - zip_ref.extractall("/content"): Extrae todos los archivos del archivo ZIP en el directorio /content de Colab.

   - print("Tiempo de ejecución para cargar y descomprimir los datos:", time.time() - start_time, "segundos"): Imprime el tiempo transcurrido para la operación de carga y descompresión de datos en segundos.

5. Este código realiza una serie de operaciones necesarias para cargar datos, definir un modelo de aprendizaje profundo, entrenar el modelo y evaluar su rendimiento en un conjunto de datos de imágenes. Cada función y parte del código contribuye a tiene un proceso en particular para la realización del proyecto:

   ```python
   def custom_transform(image):
       image = ImageOps.fit(image, (224, 224), method=0, bleed=0.0, centering=(0.5, 0.5))
       return image

   # Transformaciones de datos
   data_transforms = transforms.Compose([
       transforms.Resize((225, 225)),
       custom_transform,
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])

   def custom_loader(path):
       try:
           return Image.open(path).convert('RGB')
       except Exception as e:
           print(f"Error al cargar la imagen {path}: {e}")
           return None
   
   # Remover el archivo .DS_169665.jpg
   path_ne = os.path.join("/content/PBC_dataset_normal_DIB/neutrophil")
   os.remove(os.path.join(path_ne, ".DS_169665.jpg"))
   
   # Dataset y dataloaders
   dataset = ImageFolder("/content/PBC_dataset_normal_DIB", transform=data_transforms, loader=custom_loader)
   train_size = int(0.8 * len(dataset))
   val_size = len(dataset) - train_size
   train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
   val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

   # Modelo ViT
   model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True)
   for param in model.parameters():
       param.requires_grad = False
   num_classes = 8
   num_inputs = model.head.in_features
   last_layer = nn.Linear(num_inputs, num_classes)
   model.head = last_layer

   # Definir la función de pérdida y el optimizador
   criterion = nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

   # Entrenamiento
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print("*"*80)
   print("Se está ejecutando con GPU" if torch.cuda.is_available() else "Se está ejecutando con CPU")
   print("*"*80)
   model.to(device)

   num_epochs = 20
   train_losses = []
   val_losses = []
   for epoch in range(num_epochs):
       model.train()
       running_loss = 0.0
       for images, labels in train_loader:
           try:
               images, labels = images.to(device), labels.to(device)

               optimizer.zero_grad()
               outputs = model(images)
               loss = criterion(outputs, labels)
               loss.backward()
               optimizer.step()

               running_loss += loss.item() * images.size(0)
           except Exception as e:
               print(f"Error: {e}")

       epoch_loss = running_loss / len(train_loader.dataset)
       train_losses.append(epoch_loss)
       print("*"*80)
       print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

       # Validación
       model.eval()
       val_running_loss = 0.0
       correct = 0
       total = 0
       predictions = []
       true_labels = []
       with torch.no_grad():
           for images, labels in val_loader:
               images, labels = images.to(device), labels.to(device)
               outputs = model(images)
               loss = criterion(outputs, labels)
               val_running_loss += loss.item() * images.size(0)

               _, predicted = torch.max(outputs.data, 1)
               total += labels.size(0)
               correct += (predicted == labels).sum().item()

               predictions.extend(predicted.cpu().numpy())
               true_labels.extend(labels.cpu().numpy())

           val_loss = val_running_loss / len(val_loader.dataset)
           val_losses.append(val_loss)
           val_accuracy = correct / total
           val_balanced_accuracy = balanced_accuracy_score(true_labels, predictions)
           val_kappa = cohen_kappa_score(true_labels, predictions)

           print(f"Validation Loss: {val_loss:.4f}")
           print(f"Validation Accuracy: {val_accuracy:.4f}")
           print(f"Validation Balanced Accuracy: {val_balanced_accuracy:.4f}")
           print(f"Validation Cohen's Kappa: {val_kappa:.4f}")
           print("*"*80)

   print("Tiempo de ejecución total:", time.time() - start_time, "segundos")
   ```

   - Función custom_transform:

     - En el contexto del procesamiento de imágenes en el aprendizaje automático las transformaciones se utilizan para modificar las imágenes de entrada de alguna manera antes de que se alimenten al modelo de aprendizaje automático. Estas transformaciones pueden incluir cambios en el tamaño, la rotación, el recorte, etc.

     - La función custom_transform es una función que se utilizará como parte del proceso de transformación de imágenes antes de ser alimentadas al modelo. Esta función se define para realizar una transformación personalizada en las imágenes del conjunto de datos donde ImageOps.fit es una función de la biblioteca PIL (Python Imaging Library) que ajusta la imagen a un tamaño específico manteniendo su relación de aspecto y opcionalmente aplicando ciertos métodos de ajuste como el centrado de la imagen dentro del nuevo tamaño. En este caso, se ajusta la imagen a un tamaño de (224, 224) y al final se devuelve la imagen transformada.

     - Esto es útil como parte del proceso de preparación de datos antes de entrenar un modelo de aprendizaje automático que requiere imágenes de un tamaño uniforme como entrada.

   - Transformaciones de datos con data_transforms:

     - Aquí data_transforms define una secuencia de transformaciones que se aplicarán a las imágenes del conjunto de datos antes de ser utilizadas para entrenar el modelo. Estas transformaciones incluyen cambios en el tamaño, la orientación y el formato de las imágenes, así como la normalización de los valores de los píxeles.

     - transforms.Compose: es una función de PyTorch que se utiliza para combinar varias transformaciones en una sola secuencia de transformaciones. Se especifican las transformaciones que se aplicarán a las imágenes del conjunto de datos en el orden en que se aplicarán:

       - transforms.Resize((225, 225)): Cambia el tamaño de las imágenes a (225, 225).

       - custom_transform: Aplica la transformación personalizada definida anteriormente, en este caso, a (224, 224) píxeles mientras mantiene su relación de aspecto.

       - transforms.RandomHorizontalFlip(): Realiza un volteo horizontal aleatorio en las imágenes. Esto significa que algunas imágenes se voltearán horizontalmente con una probabilidad predeterminada, lo que puede aumentar la variabilidad de los datos y ayudar al modelo a aprender características invariantes a la orientación.

       - transforms.ToTensor(): Esta transformación convierte las imágenes PIL en tensores de PyTorch. Los modelos de PyTorch esperan tensores como entrada, por lo que esta transformación es necesaria para convertir las imágenes en un formato compatible con PyTorch. Un tensor es una estructura de datos que generaliza los conceptos de escalares, vectores y matrices a dimensiones más altas, estos se utilizan para representar tanto los datos de entrada como los parámetros del modelo. Por ejemplo, las imágenes de entrada se representan como tensores, donde las dimensiones del tensor corresponden a las dimensiones espaciales de la imagen (alto, ancho, canales de color).

       - transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]): Esta transformación normaliza los valores de los píxeles de las imágenes utilizando la media y la desviación estándar especificadas para cada canal de color (R, G, B). La normalización de los datos es una práctica común en el aprendizaje automático para asegurar que los valores de entrada estén en una escala similar, lo que puede ayudar al modelo a converger más rápido y a prevenir problemas de optimización.

   - Función custom_loader:

     - Esta función se utiliza para cargar imágenes desde archivos en una ruta específica. Utiliza la biblioteca PIL para abrir y convertir las imágenes al modo de color RGB y maneja posibles excepciones que puedan ocurrir durante el proceso de carga imprimiendo un mensaje de error. Esto garantiza que las imágenes se carguen correctamente y estén listas para ser utilizadas en el entrenamiento o la evaluación del modelo.

   - Remover el archivo .DS_169665.jpg ya que este no entrará al dataset que se utilizará por el nombre del archivo que empieza con "."

   - Creación del conjunto de datos (dataset) y los dataloaders (train_loader y val_loader):

     - Esta parte del código crea un conjunto de datos a partir de imágenes almacenadas en una estructura de directorios utilizando ImageFolder, luego lo divide en conjuntos de entrenamiento y validación con random_split con una proporción específica y define dataloaders para cargar y procesar los datos durante el entrenamiento y la evaluación del modelo.

     - Se definen los dataloaders para los conjuntos de entrenamiento y validación con el tamaño de lote especificado, y se especifica si se deben mezclar los datos y el número de subprocesos para la carga de datos. Esto asegura una preparación adecuada de los datos antes de ser utilizados para entrenar o evaluar el modelo.

   - Definir la función de pérdida y el optimizador (criterion y optimizer):

     - La función de pérdida (criterion) calcula la pérdida entre las predicciones del modelo y las etiquetas verdaderas, mientras que el optimizador (optimizer) actualiza los parámetros del modelo para minimizar esta pérdida utilizando el algoritmo Adam con una tasa de aprendizaje de 0.001. Ambos son elementos cruciales en el proceso de entrenamiento de un modelo de aprendizaje automático supervisado.

   - Entrenamiento y validación del modelo:

     - Este proceso se repite durante el número especificado de épocas (num_epochs) lo que permite al modelo mejorar gradualmente su rendimiento a medida que se ajusta a los datos de entrenamiento y se evalúa en los datos de validación.

     - Configuración del dispositivo (device):

       - Verificar si CUDA está disponible. Si CUDA está disponible device se establece en "cuda" lo que indica que se utilizará la GPU para el entrenamiento, de lo contrario, se establece en "cpu", lo que significa que se utilizará la CPU.

     - Traslado del modelo a la GPU (si está disponible):

       - Luego, el modelo se traslada al dispositivo especificado por device. Si device es "cuda", el modelo se traslada a la GPU; de lo contrario, se mantiene en la CPU. Esto asegura que el modelo y los datos se encuentren en el mismo dispositivo, lo que permite que el modelo aproveche la GPU para el procesamiento si está disponible.

     - Entrenamiento del modelo durante múltiples épocas:

       - Se inicia un bucle que recorre un número específico de épocas (num_epochs). Cada época representa una pasada completa a través de todos los datos de entrenamiento.

     - Bucle de entrenamiento:

       - Dentro del bucle de épocas, el modelo se establece en modo de entrenamiento con model.train(). Esto activa ciertos comportamientos específicos para el entrenamiento, como la activación de capas de dropout o batch normalization.

       - Se inicializa una variable running_loss para almacenar la pérdida acumulada durante el entrenamiento de cada mini-lote.

       - Se itera sobre los lotes de datos de entrenamiento (images, labels) proporcionados por el train_loader.

       - Para cada lote, se mueven las imágenes y las etiquetas al dispositivo (device) actualmente seleccionado (CPU o GPU).

       - Se establecen los gradientes del optimizador en cero con optimizer.zero_grad() para evitar la acumulación de gradientes de iteraciones anteriores.

       - Se realizan las predicciones con el modelo para obtener las salidas (outputs).

       - Se calcula la pérdida entre las salidas y las etiquetas reales con la función de pérdida (criterion).

       - Se realiza la retropropagación de la pérdida para calcular los gradientes de los parámetros del modelo con loss.backward().

       - Se actualizan los parámetros del modelo utilizando el optimizador con optimizer.step().

       - Se acumula la pérdida total de este mini-lote en running_loss.

     - Cálculo de la pérdida promedio del entrenamiento (epoch_loss):

       - Después de procesar todos los lotes de entrenamiento, se calcula la pérdida promedio del entrenamiento dividiendo la pérdida acumulada (running_loss) entre el número total de muestras en el conjunto de datos de entrenamiento.

     - Impresión de métricas de entrenamiento:

       - Se imprime la pérdida promedio del entrenamiento para esta época.

     - Validación del modelo:

       - Luego, se evalúa el modelo en el conjunto de validación.

       - Se establece el modelo en modo de evaluación con model.eval(). Esto desactiva ciertos comportamientos específicos para el entrenamiento, como la activación de capas de dropout o batch normalization.

       - Se inicializan variables para calcular la pérdida de validación (val_running_loss), el número total de predicciones correctas (correct) y otras métricas como la precisión balanceada y el kappa de Cohen.

       - Se itera sobre los lotes de datos de validación proporcionados por el val_loader.

       - Para cada lote, se mueven las imágenes y las etiquetas al dispositivo (device) actualmente seleccionado (CPU o GPU).
       - Se realizan las predicciones con el modelo para obtener las salidas (outputs).

       - Se calcula la pérdida entre las salidas y las etiquetas reales con la función de pérdida (criterion).

       - Se acumula la pérdida total de validación (val_running_loss) y se calculan otras métricas como la precisión y el kappa.

       - Finalmente, se imprime la pérdida de validación y otras métricas para esta época.

   - Medición del tiempo de ejecución total:

     - Se imprime el tiempo total que lleva ejecutar todo el código desde el inicio hasta este punto en específico.

6. Este código proporciona visualizaciones importantes para evaluar el rendimiento del modelo de clasificación incluyendo gráficas de pérdida durante el entrenamiento y la validación, una matriz de confusión y visualizaciones de resultados individuales. Estas visualizaciones ayudan a comprender cómo el modelo está aprendiendo.

   ```python
   # Gráficas
   plt.figure(figsize=(10, 5))
   plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
   plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.title('Training and Validation Loss')
   plt.legend()
   plt.show()

   # Obtener los nombres de las clases (según nombres de carpetas)
   class_names = [folder_name.split("_")[0] for folder_name in dataset.classes]

   # Matriz de confusión
   cm = confusion_matrix(true_labels, predictions)
   plt.figure(figsize=(14, 12))
   disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
   disp.plot(cmap='Blues', xticks_rotation=45, colorbar=False)
   plt.title('Confusion Matrix')
   plt.show()

   # Obtener predicciones y etiquetas en el conjunto de validación
   model.eval()
   val_predictions = []
   val_true_labels = []
   val_probabilities = []
   with torch.no_grad():
       for images, labels in val_loader:
           images, labels = images.to(device), labels.to(device)
           outputs = model(images)
           probabilities = F.softmax(outputs, dim=1)
           _, predicted = torch.max(outputs.data, 1)

           val_predictions.extend(predicted.cpu().numpy())
           val_true_labels.extend(labels.cpu().numpy())
           val_probabilities.extend(probabilities.cpu().numpy())

   # Mostrar resultados
   num_samples = 6 # controlar la cantidad de muestras visualizadas
   plt.figure(figsize=(18, 14))
   for i in range(num_samples):
       plt.subplot(2, num_samples, i + 1)
       plt.imshow(val_dataset[i][0].permute(1, 2, 0))
       plt.title(f"True: {val_true_labels[i]}, Pred: {val_predictions[i]}")
       plt.axis('off')

       plt.subplot(2, num_samples, num_samples + i + 1)
       plt.bar(range(8), val_probabilities[i])
       plt.xlabel('Class')
       plt.ylabel('Probability')
       plt.title('Predicted Probabilities')
       plt.xticks(range(8))

       # Añadir leyenda
       legend_text = '\n'.join([f'{j}: {class_names[j]}' for j in range(8)])
       plt.text(0.5, -0.3, f'Class Names:\n{legend_text}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)

   plt.tight_layout()
   plt.show()
   ```

   - Gráficas de pérdida durante el entrenamiento y la validación:

     - Se crea una nueva figura (plt.figure()) para las gráficas con un tamaño de 10x5 pulgadas.

     - Se trazan dos líneas en la misma gráfica utilizando plt.plot() para representar la pérdida durante el entrenamiento (train_losses) y la pérdida durante la validación (val_losses) a lo largo de las épocas.

     - Se etiquetan los ejes x e y con "Epoch" y "Loss" respectivamente utilizando plt.xlabel() y plt.ylabel().

     - Se establece el título de la gráfica como "Training and Validation Loss" utilizando plt.title().

     - Se añade una leyenda a la gráfica utilizando plt.legend() con etiquetas "Train Loss" y "Validation Loss".

     - Finalmente, se muestra la gráfica utilizando plt.show().

   - Matriz de confusión:

     - La matriz de confusión es una herramienta que se utiliza para evaluar el rendimiento de un modelo de clasificación. Proporciona una representación visual de las predicciones del modelo en comparación con las etiquetas verdaderas de los datos de prueba. La matriz organiza las predicciones en forma de una tabla donde cada fila representa la clase verdadera y cada columna representa la clase predicha por el modelo.

     - En una matriz de confusión típica:

       - Los elementos en la diagonal principal representan las predicciones correctas, es decir, los casos donde la clase predicha coincide con la clase verdadera.

       - Los elementos fuera de la diagonal principal representan las predicciones incorrectas, donde el modelo ha clasificado incorrectamente las muestras.

     - Se calcula la matriz de confusión (cm) utilizando la función confusion_matrix de scikit-learn, que toma las etiquetas verdaderas (true_labels) y las predicciones del modelo (predictions) en el conjunto de validación.

     - Se crea una nueva figura para la matriz de confusión con un tamaño de 14x12 pulgadas.

     - Se utiliza ConfusionMatrixDisplay para visualizar la matriz de confusión. Se especifican las etiquetas de las clases (class_names) y se selecciona un mapa de colores (en este caso, 'Blues').

     - Se establece el título de la matriz de confusión como "Confusion Matrix".

     - Finalmente, se muestra la matriz de confusión utilizando plt.show().

- Visualización de resultados individuales:

  - Se obtienen las predicciones (val_predictions), las etiquetas verdaderas (val_true_labels) y las probabilidades predichas (val_probabilities) para el conjunto de validación utilizando el modelo entrenado.

  - Se establece el número de muestras que se mostrarán (num_samples) (imágenes individuales en el conjunto de datos de validación. Cada muestra consiste en una imagen, su etiqueta verdadera y la predicción realizada por el modelo entrenado)

  - Se crea una nueva figura con un tamaño de 18x14 pulgadas.

  - Para cada muestra en el conjunto de validación, se visualiza la imagen original junto con la etiqueta verdadera y la predicción del modelo.

  - También se traza un gráfico de barras para mostrar las probabilidades predichas para cada clase.

  - Se añade una leyenda que muestra el nombre de cada clase y su correspondiente índice.

  - Se ajusta el diseño de la figura para asegurar que los elementos no se superpongan utilizando plt.tight_layout().

  - Finalmente, se muestra la figura utilizando plt.show().
