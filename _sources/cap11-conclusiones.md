# Capítulo 11 · Conclusiones y trabajo futuro

> **Objetivo:** sintetizar los principales hallazgos del proyecto *Ames Housing* y plantear posibles líneas de mejora o extensión para trabajos futuros.

---

## 11.1 Conclusiones generales

El análisis y modelado realizado a lo largo de los capítulos permitió construir un flujo completo de trabajo estadístico aplicado, desde la **limpieza de datos** hasta la **regularización avanzada de modelos lineales**.  
Entre los principales resultados se destacan:

1. **Comprensión exploratoria:** se identificaron relaciones claras entre `SalePrice` y variables estructurales como `GrLivArea`, `OverallQual`, `TotalBsmtSF` y `YearBuilt`.  
2. **Modelado base (OLS):** el modelo lineal clásico proporciona una interpretación directa, pero presenta sensibilidad a la heterocedasticidad y a la colinealidad entre predictores.  
3. **Inferencia:** los intervalos de confianza y las pruebas t mostraron significancia para las variables de calidad y área, confirmando su influencia estadística en el precio.  
4. **Diagnóstico:** se detectó ligera heterocedasticidad y algunos outliers influyentes, motivando la aplicación de métodos robustos.  
5. **Remedios:** las correcciones HC3 y los modelos RLM mejoraron la consistencia de las estimaciones, reduciendo la influencia de valores atípicos.  
6. **Regularización:** las técnicas Ridge, Lasso y ElasticNet redujeron la varianza de los estimadores, con ElasticNet logrando el mejor equilibrio entre estabilidad y parsimonia.  
7. **Validación cruzada:** confirmó la generalización de los modelos seleccionados, evitando sobreajuste y permitiendo comparar objetivamente el desempeño predictivo.

---

## 11.2 Limitaciones

A pesar de los avances logrados, el trabajo presenta algunas limitaciones que deben reconocerse:

- **Dependencia del dataset Ames Housing:** aunque es un conjunto de datos estándar, puede no reflejar condiciones reales del mercado inmobiliario actual.  
- **Supuestos lineales:** los modelos OLS y sus variantes parten de una relación lineal entre predictores y respuesta, lo cual puede simplificar excesivamente el comportamiento real.  
- **No se incluyeron variables categóricas complejas** (como vecindario o materiales detallados) en modelos de regularización, por lo que se podrían explorar codificaciones más elaboradas (OneHotEncoder).  
- **Escalado global:** las transformaciones logarítmicas o normalizaciones podrían optimizarse mediante pipelines automáticos.  
- **Ausencia de evaluación externa:** no se validó el modelo final sobre un conjunto de prueba independiente (hold-out o datos reales de predicción).  

---

## 11.3 Trabajo futuro

Como extensión del trabajo actual, se sugieren las siguientes líneas de investigación y mejora:

1. **Validación externa:** separar un conjunto de datos *test* para evaluar la capacidad predictiva real del modelo ElasticNet final.  
2. **Modelos no lineales:** explorar algoritmos de *machine learning* (Random Forests, XGBoost, SVR) y compararlos con el marco lineal.  
3. **Selección automática de hiperparámetros:** implementar búsqueda en malla (*GridSearchCV*) o Bayesian Optimization para ajustar `alpha` y `l1_ratio`.  
4. **Análisis espacial:** incluir coordenadas o vecindarios geográficos mediante modelos espaciales o regresiones geográficamente ponderadas (GWR).  
5. **Interpretabilidad avanzada:** emplear técnicas de descomposición SHAP o LIME para cuantificar el impacto marginal de cada predictor.  
6. **Publicación reproducible:** integrar el libro en un entorno web con Binder o JupyterHub para permitir la ejecución directa del código.  

---

## 11.4 Reflexión final

El estudio demuestra cómo un enfoque estadístico riguroso, apoyado en técnicas de regularización y diagnóstico, permite construir modelos explicativos robustos y replicables.  
El **Jupyter Book** elaborado consolida tanto la parte teórica como la práctica, facilitando la comprensión progresiva del proceso de análisis y modelado de datos.

> **Frase final:** *La estadística aplicada no solo busca ajustar modelos, sino entender los datos para tomar mejores decisiones.*
