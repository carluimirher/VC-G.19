# Tarea 1: Detección y conteo del valor de las monedas

Este código utiliza la cámara en tiempo real para detectar monedas, medir sus diámetros y clasificarlas según su valor en euros. Para ello, se selecciona una moneda de referencia de 1 €, que permite calcular la escala de conversión de píxeles a milímetros. Finalmente, se muestra el valor de cada moneda detectada y se calcula la suma total del dinero y se muestra por pantalla.

* **Captura de vídeo:** Utiliza `cv2.VideoCapture(0)` para obtener imágenes en tiempo real desde la cámara y procesarlas continuamente.

* **Preprocesamiento de la imagen:** Convierte cada fotograma a escala de grises mediante `cv2.cvtColor` y aplica un filtro gaussiano (`cv2.GaussianBlur`) para reducir el ruido y facilitar la detección de las monedas.

* **Umbralización de Otsu:** Emplea `cv2.threshold` con `cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU` para obtener una imagen binaria que permita separar los objetos del fondo automáticamente.

* **Detección de contornos:** Utiliza `cv2.findContours` para localizar los contornos externos de los objetos. Estos se dibujan en verde sobre la imagen del vídeo.

* **Selección de la moneda de referencia:** Al hacer clic sobre una moneda, se calcula su diámetro en píxeles mediante `cv2.minEnclosingCircle`. Esta moneda se considera de 1 €, cuyo diámetro oficial es de 23,25 mm, y se utiliza para calcular la escala de conversión a milímetros por píxel.

* **Medición y filtrado de monedas:** Una vez establecida la escala, se mide el diámetro de cada contorno detectado y se convierte a milímetros. Se descartan los objetos cuyo diámetro sea inferior al 55 % o superior al 115 % del diámetro de referencia, para reducir la detección de objetos demasiado pequeños o grandes.

* **Clasificación de las monedas:** La función `clasificar_moneda` compara el diámetro medido con los diámetros oficiales definidos en el diccionario `diametro_monedas` y selecciona el valor cuya medida sea la más cercana.

* **Visualización de resultados:** Como se ve en la siguiente imagen, se muestra el valor estimado de cada moneda en rojo y se dibuja el contorno de los objetos detectados en verde.

![Tarea 1](img/Monedas.png)

* **Cálculo del total:** El programa suma los valores de las monedas clasificadas y muestra el resultado en la parte superior izquierda de la imagen. La tecla `ESC` permite finalizar la ejecución y cerrar las ventanas.

Para esta tarea, se le preguntó a la Inteligencia Artificial ChatGPT sobre cómo se podría detectar un click dentro de uno de los contornos:
https://chatgpt.com/share/6ad90de0-f130-83ed-82e9-e888653a2903
