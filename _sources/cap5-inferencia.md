# Capítulo 5 · Inferencia y grados de libertad

> **Objetivo:** realizar **inferencia** sobre los parámetros del modelo OLS usando el dataset *Ames Housing*, mostrando el **resumen estadístico tipo tabla** (`statsmodels.OLS.summary()`), y explicitando los **grados de libertad** del modelo y de los residuos.

---

## 5.1 Definiciones y grados de libertad

Para un modelo lineal con \(n\) observaciones y \(k\) predictores (sin contar el intercepto), los grados de libertad son:

$$
Df_{\text{Model}} = k,
\qquad
Df_{\text{Residuals}} = n-k-1.
$$

Los **errores estándar** provienen de

$$
\operatorname{Var}(\hat\beta) = \hat\sigma^2 \,(X^\top X)^{-1},
\quad
\hat\sigma^2=\frac{SS_{Res}}{n-k-1},
$$

y los estadísticos **t** se calculan como

$$
t_j=\frac{\hat\beta_j}{SE(\hat\beta_j)},
\quad\text{con } t_j \sim t_{\,n-k-1}.
$$

---

## 5.2 Datos y modelo (mismas **variables candidatas** del Cap. 4)

```python
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm

# Localiza el CSV
CANDIDATE_PATHS = [Path("data/ames_housing.csv"), Path("AmesHousing.csv")]
for p in CANDIDATE_PATHS:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("No se encontró data/ames_housing.csv ni AmesHousing.csv")

df = pd.read_csv(DATA_PATH)
target = "SalePrice"

# Selección automática de variables candidatas (idéntica lógica a Cap. 4)
num_df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
corr_abs = num_df.corr(numeric_only=True)[target].dropna().abs().sort_values(ascending=False)
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

# Datos finales (sin NA) y ajuste
data = num_df[[target] + sel].dropna()
y = data[target].values
X = sm.add_constant(data[sel].values, has_constant="add")

model = sm.OLS(y, X).fit()

n = X.shape[0]
k = X.shape[1] - 1  # sin contar el intercepto
df_model = k
df_resid = n - k - 1

sel, n, k, df_model, df_resid
```
---

## 5.3 Resumen estadístico (tabla tipo `statsmodels`)

```python
model.summary()
```
> La tabla incluye: **coef**, **std err**, **t**, **P>|t|**, e **IC(95%)** por parámetro, además de métricas globales como **R²** y **F-stat**.

---

## 5.4 Intervalos de confianza (95%) y variables significativas

```python
# Intervalos de confianza (95%)
ci = model.conf_int(alpha=0.05)
ci.columns = ["CI_low", "CI_high"]

# Tabla compacta con significancia
import pandas as pd
tbl = pd.DataFrame({
    "coef": model.params,
    "std_err": model.bse,
    "t": model.tvalues,
    "p>|t|": model.pvalues
})
tbl = pd.concat([tbl, ci], axis=1)

# Marcamos significancia al 5%
tbl["signif_5%"] = np.where(tbl["p>|t|"] < 0.05, "Sí", "No")
tbl
```

---

## 5.5 Key takeaways

- El **resumen estadístico** muestra los coeficientes, sus errores estándar, pruebas t y p-values.  
- Los **grados de libertad** usados por `statsmodels` coinciden con \(Df_{\text{Model}}=k\) y \(Df_{\text{Residuals}}=n-k-1\).  
- Identificar las variables con **p-value < 0.05** permite resaltar las más influyentes; interpretar siempre su **efecto práctico** y el **intervalo de confianza** asociado.
