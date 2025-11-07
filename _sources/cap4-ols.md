# Capítulo 4 · Formulación matricial del modelo OLS

> **Objetivo:** derivar y aplicar el estimador matricial de Mínimos Cuadrados Ordinarios (OLS)  
> \[ \hat{\beta}=(X^\top X)^{-1}X^\top y \]  
> usando **las variables candidatas** seleccionadas en el Capítulo 3 y comparar con `statsmodels.OLS`.

---

## 4.1 Datos y construcción de \(X\) y \(y\)

En este capítulo construiremos \(X\) (predictores) y \(y\) (respuesta) a partir del dataset *Ames Housing*.
Tomamos como objetivo \(y = \texttt{SalePrice}\) y elegimos **hasta 12 variables candidatas** con alta correlación y baja colinealidad (según el criterio del Cap. 3).

```python
import pandas as pd
import numpy as np
from pathlib import Path

# Localiza el CSV (compatible con tu estructura de proyecto)
CANDIDATE_PATHS = [Path("data/ames_housing.csv"), Path("AmesHousing.csv")]
for p in CANDIDATE_PATHS:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("No se encontró data/ames_housing.csv ni AmesHousing.csv")

df = pd.read_csv(DATA_PATH)

target = "SalePrice"

# Asegurar numérico y eliminar infinitos
num_df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

# Pool por correlación absoluta con el target
corr_abs = num_df.corr(numeric_only=True)[target].dropna().abs().sort_values(ascending=False)

# Empezamos con top 15 (excluyendo el propio SalePrice) y filtramos por colinealidad > 0.85
pool = [c for c in corr_abs.index if c != target][:15]

sel = []
for v in pool:
    if not sel:
        sel.append(v)
        continue
    ok = True
    for u in sel:
        r = abs(num_df[[v, u]].dropna().corr().iloc[0,1])
        if r > 0.85:
            ok = False
            break
    if ok:
        sel.append(v)
    if len(sel) >= 12:
        break

# Construcción de matrices con intercepto
data = num_df[[target] + sel].dropna()
y = data[target].values.reshape(-1, 1)
X = data[sel].values
X = np.c_[np.ones((X.shape[0], 1)), X]  # intercepto

sel, X.shape, y.shape
```
El vector de variables elegidas `sel` es el conjunto de **candidatas** para este capítulo.  

---

## 4.2 Derivación del estimador OLS

Sea el problema de mínimos cuadrados: minimizar \(S(\beta)=\|y-X\beta\|^2\).  
La condición de primer orden (ecuaciones normales) es
\[ X^\top X\,\hat{\beta} = X^\top y. \]
Si \(X^\top X\) es invertible (pleno rango), entonces
\[ \hat{\beta}=(X^\top X)^{-1}X^\top y. \]

> **Invertibilidad:** Se requiere \(\operatorname{rango}(X)=p\) (columnas linealmente independientes).  
> Con intercepto, \(p=\) número de predictores + 1.

---

## 4.3 Cálculo numérico manual

```python
# Cálculo manual de beta_hat
XtX = X.T @ X
Xty = X.T @ y

# Comprobaciones de rango y número de condición
rank = np.linalg.matrix_rank(XtX)
cond = np.linalg.cond(XtX)

beta_hat = np.linalg.inv(XtX) @ Xty

rank, cond, beta_hat[:5].ravel()  # mostramos primeras 5 betas
```

### Predicciones, residuos y suma de cuadrados

```python
yhat = X @ beta_hat
res  = y - yhat
SSR  = float((res.T @ res))  # residual sum of squares
n, p = X.shape
sigma2_hat = SSR / (n - p)

n, p, SSR, sigma2_hat
```

### Varianzas de \(\hat{\beta}\) y errores estándar

```python
# Var(beta_hat) = sigma^2 * (X'X)^{-1}
var_beta = sigma2_hat * np.linalg.inv(XtX)
se_beta = np.sqrt(np.diag(var_beta)).reshape(-1,1)

se_beta[:5].ravel()
```

---

## 4.4 Comparación con `statsmodels.OLS`

```python
import statsmodels.api as sm

# Con statsmodels (agrega intercepto)
y_sm = data[target].values
X_sm = sm.add_constant(data[sel].values, has_constant="add")

ols = sm.OLS(y_sm, X_sm).fit()

print(ols.summary())
```

> **Coincidencia:** Los coeficientes, errores estándar y métricas (R², SSR) deben coincidir con los obtenidos manualmente (salvo diferencias de redondeo).  
> Verifica especialmente `params` y `bse` frente a `beta_hat` y `se_beta` calculados arriba.

---

## 4.5 Discusión: invertibilidad y significado de cada término

- \(X^\top X\) **singular** → ocurre con multicolinealidad perfecta (columnas duplicadas o combinación lineal exacta).  
- \( \kappa(X^\top X) = \text{condición} \) **alta** → problemas de inestabilidad numérica; considerar eliminar predictores redundantes o regularización.  
- **Interpretación:**  
  - Intercepto: precio esperado cuando los predictores valen cero (interpretar con cautela).  
  - Pendientes: cambio esperado en `SalePrice` por unidad del predictor, manteniendo los demás constantes.

> Para diagnóstico adicional, considera calcular **VIF** y estandarizar predictores antes del ajuste.

---

## 4.6 Key takeaways

- El estimador matricial \( (X^\top X)^{-1}X^\top y \) coincide con `statsmodels.OLS`.  
- La **invertibilidad** depende del rango de \(X\); la colinealidad alta puede inflar varianzas y volver inestable la estimación.  
- Usar variables **candidatas** reduce colinealidad y mejora interpretabilidad del modelo base que se ampliará en los siguientes capítulos.
