# Capítulo 1 · Introducción y pregunta de investigación

> **Overview:**  
> Este capítulo introduce el proyecto integrador, contextualiza el conjunto de datos *Ames Housing* y plantea las preguntas de investigación y objetivos que guían el análisis posterior.

---

## 1.1 Contexto del dataset

El conjunto de datos **Ames Housing** fue compilado por *Dean De Cock (2011)* como una alternativa moderna al clásico *Boston Housing Dataset*.  
Contiene información detallada sobre **2930 viviendas** en la ciudad de *Ames, Iowa (EE.UU.)*, con **82 variables** que describen características estructurales, de ubicación y de calidad de las propiedades.

Entre sus variables más importantes se encuentran:

- `SalePrice`: precio de venta de la vivienda (variable respuesta principal).  
- `GrLivArea`: área habitable sobre el suelo.  
- `OverallQual`: calidad general del material y acabado.  
- `GarageCars`, `TotalBsmtSF`, `YearBuilt`, entre otras variables predictoras.

---

### Resumen estadístico inicial

A continuación se muestra un resumen estadístico de las primeras variables del conjunto de datos:

| Variable | Conteo | Media | Desv. Est. | Mínimo | Máximo |
|-----------|--------|--------|-------------|---------|---------|
| Order | 2930 | 1465.50 | 845.96 | 1.00 | 2930.00 |
| PID | 2930 | 714464496.99 | 188730844.65 | 526301100.00 | 1007100110.00 |
| MS SubClass | 2930 | 57.39 | 42.64 | 20.00 | 190.00 |
| Lot Frontage | 2440 | 69.22 | 23.37 | 21.00 | 313.00 |
| Lot Area | 2930 | 10147.92 | 7880.02 | 1300.00 | 215245.00 |


---

## 1.2 Preguntas de investigación

A partir de este contexto, se proponen las siguientes preguntas de investigación:

1. **¿Qué variables estructurales influyen más en el precio de venta (`SalePrice`) de las viviendas en Ames, Iowa?**

2. **¿Qué tan estables son los coeficientes estimados bajo métodos de regresión robusta frente a los del modelo clásico de mínimos cuadrados ordinarios (OLS)?**

Estas preguntas permitirán evaluar tanto el poder explicativo del modelo lineal como su **sensibilidad ante violaciones de los supuestos clásicos**.

---

## 1.3 Objetivos específicos

- Analizar la relación entre las variables predictoras más relevantes y el precio de venta (`SalePrice`).  
- Comparar los resultados de la regresión lineal clásica con los obtenidos mediante **métodos robustos** (por ejemplo, M-estimadores).  
- Evaluar la estabilidad de los coeficientes bajo diferentes estrategias de validación y diagnóstico de supuestos.  
- Garantizar la **reproducibilidad** del análisis mediante el uso de un entorno controlado (`requirements.txt` o `environment.yml`).

---

## 1.4 Criterios de éxito

- Los modelos deben cumplir con los supuestos de **linealidad**, **homocedasticidad** y **normalidad de residuos**, o justificar su incumplimiento mediante métodos robustos.  
- Todos los análisis deben ser **reproducibles** y acompañarse de interpretación estadística.  
- La validación cruzada o el diagnóstico gráfico deben sustentar las conclusiones del modelo.  
- El capítulo debe cerrar con una sección **“Key takeaways”** que resuma los hallazgos principales.

---

## Key takeaways

- El dataset *Ames Housing* ofrece una base rica para estudiar relaciones entre variables estructurales y el precio de venta de viviendas.  
- Las preguntas de investigación se centran en identificar las variables más influyentes y en evaluar la robustez de los coeficientes del modelo.  
- Se establecerán criterios claros de reproducibilidad e interpretación para garantizar la validez del análisis estadístico.
