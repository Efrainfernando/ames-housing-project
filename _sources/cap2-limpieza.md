# Capítulo 2 · Descripción y limpieza del dataset

> **Overview:**  
> Este capítulo describe el proceso de inspección, limpieza y preparación del conjunto de datos *Ames Housing*.  
> Se presentan sus características generales, el tratamiento de valores faltantes y la codificación de variables categóricas antes del modelado.

---

## 2.1 Fuente y estructura del dataset

- **Fuente:** Dataset público *Ames Housing* disponible en [Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset).  
- **Tamaño:** 2930 registros y 82 variables.  
- **Licencia:** Dominio público para uso educativo y académico.  
- **Variables:** Incluye tanto variables numéricas como categóricas relacionadas con aspectos estructurales, de calidad, ubicación y año de construcción.

---

## 2.2 Tabla de tipos de variables y valores faltantes

Se elabora una tabla con el tipo de variable (numérica o categórica) y el porcentaje de valores faltantes por columna.  
Ejemplo:

| Variable       | Tipo       | % Faltantes |
|----------------|-------------|--------------|
| LotFrontage    | Numérica    | 17.7% |
| Alley          | Categórica  | 93.2% |
| MasVnrArea     | Numérica    | 0.5% |
| Electrical     | Categórica  | 0.1% |

> 💡 *Interpretación:* Variables como `Alley` presentan gran cantidad de valores faltantes, por lo que se deben evaluar estrategias de imputación o exclusión.

---

## 2.3 Manejo de valores faltantes

Los valores faltantes se tratan según el tipo de variable:

- **Numéricas:** Imputación con la **mediana** o mediante regresión simple.  
- **Categóricas:** Imputación con la **moda** o asignación de una categoría “No aplica”.  
- **Altamente faltantes (>90%)**: Eliminación si su aporte informativo es bajo.

---

## 2.4 Detección y tratamiento de outliers

Los valores atípicos se detectan mediante:
- Diagramas de caja y bigotes (Boxplots).  
- Regla de 1.5×IQR (Rango intercuartílico).  
- Comparación con los valores esperados del modelo OLS inicial.

> 🔍 *Ejemplo:* `GrLivArea` y `SalePrice` suelen contener outliers asociados a casas de lujo o construcciones no típicas.

Los outliers se pueden:
- Reemplazar por límites truncados.  
- Ajustar mediante **transformaciones logarítmicas**.  
- O mantener, si representan información relevante (p. ej., viviendas de alto valor).

---

## 2.5 Transformaciones y codificación

- **Transformaciones logarítmicas:**  
  Se aplican a variables sesgadas como `SalePrice`, `GrLivArea` y `LotArea` para mejorar la normalidad de los residuos.

- **Codificación de variables categóricas:**  
  - *One-Hot Encoding* para variables nominales.  
  - *Ordinal Encoding* para variables con jerarquía, como `OverallQual` o `ExterCond`.

---

## 2.6 Entregable

El entregable de este capítulo consiste en una **tabla comparativa “antes y después”** del proceso de limpieza, mostrando:
- Número de observaciones y variables.  
- Porcentaje total de valores faltantes.  
- Número de outliers detectados y tratados.  
- Transformaciones aplicadas y justificación.

---

## Key takeaways

- La limpieza de datos es esencial para asegurar la validez de los modelos de regresión.  
- Las imputaciones deben documentarse y justificarse.  
- Las transformaciones logarítmicas y codificaciones categóricas mejoran la interpretación y estabilidad del modelo.  
- El capítulo concluye con un dataset limpio, listo para análisis exploratorio y modelado.