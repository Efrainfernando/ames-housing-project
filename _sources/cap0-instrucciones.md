# Capítulo 0 · Instrucciones de reproducción y estructura del proyecto

> **Overview:**  
> Este capítulo presenta las instrucciones para compilar, reproducir y estructurar correctamente el libro del proyecto integrador *“Inferencia robusta y validación en modelos de regresión lineal usando el Ames Housing Dataset”*.  
> Se detallan los requerimientos técnicos, control de versiones y lineamientos de formato para todos los capítulos del libro.

---

## 0.1 Cómo compilar el libro

Ejecutar el siguiente comando en la terminal dentro de la carpeta raíz del proyecto:

```bash
jupyter-book build .
```

Los archivos compilados se generan en la carpeta `_build/html/`.

### Dependencias

Instalar las librerías listadas en el archivo `requirements.txt` o alternativamente usar un entorno con `environment.yml`:

```bash
pip install -r requirements.txt
```

---

## 0.2 Cómo obtener el dataset

El libro utiliza el conjunto de datos **Ames Housing**.  
Se espera en la ruta:

```
data/ames_housing.csv
```

Puedes descargarlo manualmente o empleando la **Kaggle API**:

```bash
kaggle datasets download -d prevek18/ames-housing-dataset -p data/ --unzip
mv data/AmesHousing.csv data/ames_housing.csv
```

---

## 0.3 Control de versiones

Estructura sugerida del proyecto:

```bash
book/
├── data/
│   └── ames_housing.csv
├── notebooks/
├── _build/
├── _config.yml
├── _toc.yml
└── requirements.txt
```

> 🔧 **Consejo:** usa control de versiones con Git y sincroniza los cambios del libro en GitHub antes de publicar.

---

## 0.4 Reproducibilidad

- Todos los experimentos deben fijar una **semilla aleatoria** (`random_state`) para garantizar resultados reproducibles.  
- Se recomienda usar Python ≥ 3.10 y registrar las versiones principales de librerías (`numpy`, `pandas`, `statsmodels`, `matplotlib`, `scikit-learn`).  
- Cada notebook debe incluir celdas comentadas para permitir su ejecución desde cero sin errores.

---

## 0.5 Requisitos técnicos del entregable

1. **El Jupyter Book debe compilar sin errores:**

```bash
jupyter-book build .
```

2. **El libro debe incluir texto interpretativo y conclusiones.**  
   Ninguna figura, tabla o ecuación debe quedar sin análisis.

3. **Todas las figuras, tablas y ecuaciones deben:**
   - Estar numeradas.  
   - Tener referencia explícita en el texto.  
   - Ser citadas en formato académico (por ejemplo, «ver Figura 3.2»).

4. **Debe incluir un archivo de entorno:**

```bash
requirements.txt
```
   o alternativamente  
```bash
environment.yml
```

   Este archivo debe especificar la versión de Python y las librerías principales utilizadas.

5. **El libro debe estar publicado correctamente en GitHub Pages** mediante:

```bash
ghp-import -n -p -f _build/html
```

📘 *Sugerencia:* Antes de publicar, verifica que las rutas de imágenes, notebooks y datos sean relativas (por ejemplo, `../data/archivo.csv`)  
y que la carpeta `_build/html` se genere sin advertencias.

---

## 0.6 Estructura general del proyecto

Cada capítulo del libro debe iniciar con un **Resumen (overview)** de 3–5 líneas que explique brevemente su propósito y contenido,  
y finalizar con una sección **«Key takeaways»** que sintetice los aprendizajes principales.

---

### 0.6.1 Capítulo 0: Instrucciones de reproducción

1. **Cómo compilar el libro:**

```bash
jupyter-book build .
```

   - Dependencias:  
     - `requirements.txt`  
     - o `environment.yml`

2. **Cómo obtener el dataset:**
   - Ruta esperada: `data/ames_housing.csv`
   - Descarga manual o usando la **Kaggle API**:

```bash
kaggle datasets download -d prevek18/ames-housing-dataset -p data/ --unzip
mv data/AmesHousing.csv data/ames_housing.csv
```

3. **Control de versiones:**

```bash
book/
├── data/
├── notebooks/
├── _build/
└── _config.yml
```

4. **Semillas reproducibles:**  
   Incluir el parámetro `random_state` en todos los experimentos.

5. **Mapa del libro:**  
   Incluir la lista de capítulos con enlaces internos a cada sección.

---

## 0.7 Key takeaways

- El libro debe ser **completamente reproducible** y **compilar sin errores**.  
- Se exige una estructura clara, con capítulos bien documentados y conclusiones interpretativas.  
- Las figuras, tablas y ecuaciones deben integrarse dentro del texto con análisis contextual.  
- La publicación final debe realizarse en **GitHub Pages** de forma funcional y accesible.  
- Este capítulo sirve como **guía técnica y metodológica** para el desarrollo del proyecto completo.

---
