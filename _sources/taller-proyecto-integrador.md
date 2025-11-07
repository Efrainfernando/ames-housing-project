# Proyecto integrador: Inferencia robusta y validación en modelos de regresión lineal usando el *Ames Housing Dataset*

## 2.11.1. Demostraciones solicitadas

1. Sea un modelo de regresión lineal simple

$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \quad i = 1, \ldots, n
$$

donde los errores aleatorios cumplen que

$$
\mathbb{E}(\varepsilon_i) = 0 \quad \text{y} \quad \text{Var}(\varepsilon_i) = \sigma^2.
$$

Demuestra que la suma de cuadrados de los residuos dividida por $\sigma^2$:

$$
\frac{SS_{Res}}{\sigma^2} = \frac{\sum_{i=1}^{n} e_i^2}{\sigma^2}
$$

puede expresarse como una **combinación cuadrática de los errores** $\varepsilon_i$.

---

### Demostración

Sabemos que el vector de errores $\varepsilon$ se puede escribir como:

$$
\varepsilon = (I - H)y
$$

donde $H = X(X'X)^{-1}X'$ es la **matriz sombrero** (*hat matrix*).

Por tanto, los residuos son:

$$
e = (I - H)\varepsilon
$$

y la suma de cuadrados de los residuos:

$$
SS_{Res} = e'e = \varepsilon'(I - H)\varepsilon.
$$

Dividiendo por $\sigma^2$:

$$
\frac{SS_{Res}}{\sigma^2} = \frac{\varepsilon'(I - H)\varepsilon}{\sigma^2}.
$$

El término $(I - H)$ es **idempotente y simétrica**, y su rango es $n - p$ donde $p$ es el número de parámetros estimados (en este caso $p=2$).  
Por propiedades de combinaciones cuadráticas, se tiene que:

$$
\frac{SS_{Res}}{\sigma^2} \sim \chi^2_{n-p}.
$$

En el modelo de regresión lineal simple, $p = 2$, por lo tanto:

$$
\frac{SS_{Res}}{\sigma^2} \sim \chi^2_{n-2}.
$$

---

### Interpretación

Los **dos grados de libertad** que se restan corresponden a los parámetros estimados $\beta_0$ y $\beta_1$.  
Esto refleja que cada parámetro ajustado “consume” un grado de libertad en la estimación de los residuos.

---

**Referencia:**  
Draper, N. R., & Smith, H. (1998). *Applied Regression Analysis.* Wiley.