# Tarea 1: Ajedrez

Este es el código genera una imagen de 800x800 píxeles con la textura de un tablero de ajedrez. Primero se observa que el código que nosotros desarrollamos, generamos una imagen de 800x800 píxeles. Y, mediante bucles, generamos celdas intercaladas de 100x100 píxeles. Mientras que la IA, fue mucho más eficiente, solo creó una matriz de 8x8 y asignó las celdas aprovechando la vectorización de NumPy, tras ello, solo tuvo que redimensionar cada casilla de un 1x1 a un 100x100, y tras visualizarlo con una escala de grises, se obtiene el tablero de ajedrez de 800x800 de una forma mucho más eficiente.

# Tarea 2: Mondrian

El siguiente código, genera una imagen creada al estilo Mondrian. Simplemente, hemos utilizado los métodos proporcionados en el ejemplo para crear dicha imagen a mano. Y al final, guarda la imagen en el disco.

# Tarea 3: Detección de Intensidad

El código rastrea los puntos extremos de iluminación del frame capturado por la cámara mediante los siguientes pasos técnicos:

* *Conversión a luminancia:* Transforma el fotograma a escala de grises (`cv2.cvtColor`), reduciendo la imagen a un solo canal de intensidades (de 0 para negro absoluto a 255 para blanco puro).
* *Búsqueda de extremos:* Utiliza la función `cv2.minMaxLoc(gray)`, que escanea la matriz de la imagen para extraer simultáneamente los valores mínimos y máximos de brillo junto con sus coordenadas exactas en píxeles (`min_loc` y `max_loc`).
* *Marcado visual dinámico:* Sobre el fotograma original en color, dibuja dos círculos delimitadores (`cv2.circle`) y etiquetas de texto (`cv2.putText`): un circulo negro en la zona de sombra más profunda y uno blanco en el punto de mayor brillo o saturación de luz.

# Tarea 4: Pop art

El código captura el vídeo en tiempo real y transforma la imagen en una ilustración estilo serigrafía pop art mediante tres pasos clave:

* *Posterización por luminosidad:* Convierte la imagen a escala de grises y divide el brillo en 3 tonos planos utilizando colores primarios en formato BGR: amarillo para las luces altas (>170), rojo para los tonos medios (85–170) y azul para las sombras (<85).
* *Simplificación de texturas:* Aplica un filtro de mediana (`cv2.medianBlur`) para eliminar el ruido y los detalles finos, logrando manchas de color homogéneas y sólidas.
* *Trazado de bordes tipo cómic:* Mediante umbralizado adaptativo (`cv2.adaptiveThreshold`), detecta los contornos de la escena en negro y los fusiona (`cv2.bitwise_and`) sobre los colores planos para delimitar las figuras con líneas marcadas.

##### Fuente: https://galeriabat.com/es/shop/artista/andy-warhol
---

# Dudas resueltas con IA

## Consulta 1: Encontrar valores extremos en escala de grises

> **Pregunta:** ¿Con qué función se puede encontrar el valor máximo y mínimo en escala de grises con OpenCV en Python?

### Introducción

En el procesamiento de imágenes digitales con OpenCV (`cv2`) en Python, determinar la intensidad lumínica más baja y más alta dentro de una matriz en escala de grises es una tarea recurrente para tareas de contraste, normalización y umbralización.

### 1. Función principal: `cv2.minMaxLoc()`

La función estándar provista por OpenCV es `cv2.minMaxLoc()`. Esta función analiza matrices de un solo canal y devuelve cuatro parámetros:

* `min_val`: Valor mínimo de intensidad de píxel.
* `max_val`: Valor máximo de intensidad de píxel.
* `min_loc`: Coordenada $(x, y)$ del primer píxel con el valor mínimo.
* `max_loc`: Coordenada $(x, y)$ del primer píxel con el valor máximo.

#### Código de ejemplo

```{python}
import cv2

# 1. Cargar la imagen directamente en escala de grises
imagen_gris = cv2.imread("imagen_ejemplo.jpg", cv2.IMREAD_GRAYSCALE)

# Verificar que la imagen se haya cargado correctamente
if imagen_gris is None:
    raise FileNotFoundError("No se encontró el archivo de imagen.")

# 2. Calcular los valores extremos y sus coordenadas
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imagen_gris)

# 3. Mostrar los resultados
print(f"Valor mínimo: {min_val} en la posición (x, y): {min_loc}")
print(f"Valor máximo: {max_val} en la posición (x, y): {max_loc}")
```

---

## Consulta 2: Uso de blur, adaptiveThreshold y bitwise_and

> **Pregunta:** Explícame el uso de las funciones `meanBlur`, `adaptiveThreshold` y `bitwise_and`.

En OpenCV y procesamiento digital de imágenes, estas tres funciones suelen formar parte de una misma tubería (*pipeline*): **`blur`** (o filtrado medio) suaviza la imagen reduciendo ruido, **`adaptiveThreshold`** la binariza adaptándose a variaciones de iluminación, y **`bitwise_and`** aplica operaciones lógicas a nivel de bits (como aislar regiones mediante máscaras).

### 1. `cv2.blur()` / Filtrado por media (`meanBlur`)

*Nota técnica:* En la API de OpenCV para Python/C++, el filtro de media estándar se invoca mediante `cv2.blur()`, mientras que la función C++ subyacente se denomina `cv::blur` (o `cv::boxFilter`).

#### ¿Qué hace?
Reemplaza el valor de cada píxel por el **promedio aritmético** de los píxeles contenidos dentro de una ventana o vecindad cuadrada/rectangular (el núcleo o *kernel*).

#### ¿Para qué sirve?
* Reducir ruido de alta frecuencia (grano suave).
* Desenfocar fondos o suavizar bordes antes de aplicar un umbral.

```python
# Sintaxis
dst = cv2.blur(src, ksize)
```

* `src`: Imagen de entrada.
* `ksize`: Tupla `(ancho, alto)` que define el tamaño del kernel (por ejemplo, `(5, 5)`). Cuanto mayor sea el número, mayor será el desenfoque.

---

### 2. `cv2.adaptiveThreshold()` (Umbralización adaptativa)

#### ¿Qué hace?
Convierte una imagen en escala de grises en una imagen binaria (blanco y negro puro). A diferencia de `cv2.threshold()`, que usa un único valor global de corte, la técnica adaptativa **calcula un umbral distinto para cada píxel** basándose en la media o suma ponderada gaussiana de su vecindad local.

#### ¿Para qué sirve?
* Segmentar documentos escaneados con sombras, viñeteado o iluminación desigual.
* Extraer texto o bordes donde un umbral fijo deja zonas completamente negras o quemadas.

```python
# Sintaxis
dst = cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
```

* `src`: Imagen de entrada de 8 bits en escala de grises.
* `maxValue`: Valor asignado si la condición del umbral se cumple (típicamente `255`).
* `adaptiveMethod`:
  * `cv2.ADAPTIVE_THRESH_MEAN_C`: El umbral es la media de la vecindad menos `C`.
  * `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`: El umbral es la suma ponderada gaussiana de la vecindad menos `C`.
* `thresholdType`: Tipo de binarización (`cv2.THRESH_BINARY` o `cv2.THRESH_BINARY_INV`).
* `blockSize`: Tamaño del vecindario local para calcular el umbral (debe ser un número impar mayor que 1, p. ej., `11`, `15`, `31`).
* `C`: Constante que se resta a la media/ponderación calculada (controla la sensibilidad frente al ruido de fondo).

---

### 3. `cv2.bitwise_and()` (Y lógico a nivel de bits)

#### ¿Qué hace?
Realiza una operación lógica `AND` bit a bit entre dos imágenes o entre una imagen y sí misma con una **máscara**. 

Si un píxel en la máscara vale `255` (todos los bits en `1`), el píxel original pasa intacto ($X \text{ AND } 1 = X$). Si el píxel en la máscara vale `0`, el resultado es negro puro ($X \text{ AND } 0 = 0$).

#### ¿Para qué sirve?
* Recortar una región de interés (ROI) conservando los colores o texturas originales.
* Aplicar el resultado de una binarización para ocultar el fondo no deseado.

```python
# Sintaxis típica para aplicar una máscara
dst = cv2.bitwise_and(src1, src2, mask=mask)
```

* `src1`, `src2`: Imágenes de entrada (deben tener idéntico tamaño y profundidad). Si se quiere recortar `src1` con una máscara, se pasa `src1` en ambos argumentos.
* `mask` *(opcional)*: Imagen binaria de 8 bits y 1 canal que determina qué píxeles se procesan.

---

### Ejemplo práctico integrando las tres funciones

Un caso de uso común: aislar el texto de una página con mala iluminación conservando los píxeles originales.

```{python}
import cv2

# 1. Cargar imagen y convertir a escala de grises
img = cv2.imread("documento.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Suavizar para eliminar grano fino que ensucie la binarización
blurred = cv2.blur(gray, (5, 5))

# 3. Binarización adaptativa: detecta texto incluso bajo sombras
# Usamos THRESH_BINARY_INV para que el texto sea blanco (255) y el fondo negro (0)
mask = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    blockSize=15,
    C=4
)

# 4. bitwise_and: extrae los píxeles de color originales del texto usando la máscara
resultado = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("Mascara", mask)
cv2.imshow("Texto Extraido", resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
