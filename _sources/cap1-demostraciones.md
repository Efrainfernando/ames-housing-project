# Capítulo 1 · Demostraciones solicitadas

> **Overview:**  
> En este capítulo se presentan las demostraciones teóricas fundamentales del modelo de **regresión lineal simple**, enfocadas en la distribución de la suma de cuadrados de los residuos y el origen de los grados de libertad asociados.

---

## 1.1 Modelo de regresión lineal simple

Sea el modelo
$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \quad i=1,\ldots,n,
$$
donde los errores cumplen:
$$
\mathbb{E}(\varepsilon_i)=0, \qquad \operatorname{Var}(\varepsilon_i)=\sigma^2.
$$

Definimos los **residuos**:
$$
e_i = y_i - \hat{y}_i = y_i - (\hat{\beta}_0+\hat{\beta}_1 x_i),
$$
y la **suma de cuadrados de los residuos**:
$$
SS_{Res} = \sum_{i=1}^n e_i^2.
$$

---

## 1.2 Objetivo de la demostración

Demostrar que:
$$
\frac{SS_{Res}}{\sigma^2} \sim \chi^2_{n-2},
$$
y explicar por qué se restan **dos grados de libertad** en el modelo de regresión simple (asociados a $\hat\beta_0$ y $\hat\beta_1$).

---

## 1.3 Marco teórico

En notación matricial:
$$
y = X\beta + \varepsilon, \qquad X = 
\begin{bmatrix}
1 & x_1\\
1 & x_2\\
\vdots & \vdots\\
1 & x_n
\end{bmatrix}.
$$

El estimador de mínimos cuadrados ordinarios (MCO) es:
$$
\hat\beta = (X'X)^{-1}X'y,
$$
y las predicciones se obtienen mediante la **matriz sombrero** $H = X(X'X)^{-1}X'$:
$$
\hat{y} = Hy, \qquad e = y - \hat{y} = (I-H)y.
$$

---

## 1.4 Demostración paso a paso

1. **Expresión de los residuos en función de los errores**  

   Sustituyendo $y = X\beta + \varepsilon$:
   $$
   e = (I-H)(X\beta+\varepsilon) = (I-H)\varepsilon.
   $$

2. **Suma de cuadrados de los residuos**
   $$
   SS_{Res} = e'e = \varepsilon'(I-H)\varepsilon.
   $$

3. **Propiedades clave de la matriz $(I-H)$:**
   - Es **simétrica**: $(I-H)' = I-H$,
   - Es **idempotente**: $(I-H)^2 = I-H$,
   - Su rango es $n - p$, donde $p$ es el número de parámetros estimados.

   En el modelo simple, $p = 2$ (intercepto y pendiente), por tanto $\operatorname{rango}(I-H) = n-2$.

4. **Distribución del cuadrático:**

   Como $\varepsilon \sim \mathcal N(0,\sigma^2 I)$,
   $$
   \frac{\varepsilon'(I-H)\varepsilon}{\sigma^2} \sim \chi^2_{n-2}.
   $$

   Por lo tanto:
   $$
   \boxed{\displaystyle \frac{SS_{Res}}{\sigma^2} \sim \chi^2_{n-2}}.
   $$

---

## 1.5 Interpretación

Los **grados de libertad** reflejan las restricciones impuestas por los parámetros estimados:
- Cada parámetro estimado ($\beta_0$, $\beta_1$) “consume” un grado de libertad.  
- En total se restan 2 grados de libertad al número de observaciones $n$.  

Por tanto, la estimación insesgada de la varianza del error es:
$$
s^2 = \frac{SS_{Res}}{n-2}.
$$

---

## 1.6 Key takeaways

- $H = X(X'X)^{-1}X'$ es fundamental para expresar los residuos y entender su estructura.  
- $(I-H)$ proyecta los errores sobre el subespacio ortogonal a $X$.  
- El estadístico $\tfrac{SS_{Res}}{\sigma^2}$ sigue una distribución $\chi^2_{n-2}$.  
- Los dos grados de libertad perdidos corresponden a los parámetros estimados.  
- Este resultado sustenta la inferencia sobre $\sigma^2$ y los test t en regresión.

---

📚 **Referencia:**  
Draper, N. R., & Smith, H. (1998). *Applied Regression Analysis.* Wiley.
