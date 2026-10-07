# Tarea 1: Píxeles blancos por fila

Este código analiza la densidad de píxeles que detecta el algoritmo de Canny para identificar las filas con mayor concentración de bordes y representarlas gráficamente.

* **Canny:** Usa este detector de contornos para crear una copia de la imagen original donde solo se dejan los bordes.
* **Obtención de filas:** Utiliza la función `cv2.reduce` sobre la imagen binarizada `canny` para sumar los valores de los píxeles a lo largo de cada fila. Para luego normalizar los recuentos y así obtener la proporción de píxeles blancos por fila.
* **Imagen Canny:** Como se ve en la imagen, se pinta sobre ella unas líneas horizontales (`plt.axhline`) en las filas que representan las filas donde la proporción de píxeles blancos supera el 90% del máximo (`filas.max() * 0.90`).

![Tarea 1](img/output1_Canny.png)

* **Perfil de densidad:** En esta gráfica de línea, se representa el porcentaje de píxeles blancos por fila, y se señalan con líneas verticales rojas todas aquellas que superan el objetivo.

![Tarea 1](img/output1_Histo.png)

###

# Tarea 2: Canny y Sobel umbralizados

El siguiente código realiza una binarización mediante umbralizado sobre los resultados de los detectores de bordes Sobel y Canny, para luego analizar la densidad de píxeles tanto por filas como por columnas, tras ello, se comparan las respuestas de ambos métodos.

* **Umbralización binaria:** Se aplica la función `cv2.threshold` sobre las imágenes `sobel8` y `canny` utilizando un `valorUmbral` para obtener las máscaras binarias de bordes (`imagenUmbralizadaSobel` e `imagenUmbralizadaCanny`).
* **Cálculo de perfiles de densidad:** Suma las intensidades de los píxeles a lo largo de las filas y de las columnas. Posteriormente, normaliza estos recuentos dividiendo entre el número total de píxeles en esa dirección multiplicados por 255.
* **Superposición visual y trazado de ejes:** Como se ve en la imagen, se muestran las binarizaciones marcando con líneas rojas aquellas filas y columnas cuya densidad de bordes supera el 90% del valor máximo en sus respectivas orientaciones.

![Tarea 1](img/output2_Umbra.png)

* **Análisis por histogramas/perfiles:** Se genera los siguientes gráficos de línea de para comparar el % de píxeles activos por cada fila y columna entre los algoritmos de Sobel y Canny.

![Tarea 1](img/output2_Histo.png)

### Comparación entre Sobel y Canny

* **Detección de textura y ruido:** Canny detecta muchos bordes finos en el pelaje, mientras que Sobel filtra mejor el ruido y se centra en los contrastes más marcados.
* **Marcado de filas y columnas:** En Sobel, las líneas se concentran solo en zonas muy concretas (como las cejas y la nariz). En Canny, la acumulación de bordes en el pelaje hace que varias columnas y filas superen el umbral.
* **Grosor del borde:** Sobel produce bordes más gruesos y difusos, mientras que Canny los afina a un único píxel, generando un mapa de contornos más detallado pero más saturado.
