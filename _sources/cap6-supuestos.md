# Capítulo 6 · Diagnóstico de supuestos

> **Objetivo:** verificar los supuestos clásicos del modelo OLS ajustado (mismas **variables candidatas** de los capítulos 4–5) y presentar una **tabla resumen** con decisiones.

---

## 6.1 Ajuste del modelo (reutilizando lógica del Cap. 5)

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

# Selección automática de variables candidatas (idéntica a Cap. 4–5)
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

data = num_df[[target] + sel].dropna()
y = data[target].values
X = sm.add_constant(data[sel].values, has_constant="add")

model = sm.OLS(y, X).fit()
sel, X.shape, model.rsquared
```
---

## 6.2 Linealidad: Residual vs Fitted

```python
import matplotlib.pyplot as plt

fitted = model.fittedvalues
resid  = model.resid

plt.figure(figsize=(6,4))
plt.scatter(fitted, resid, s=12, alpha=0.6)
plt.axhline(0, linestyle='--')
plt.title('Residual vs Fitted')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.show()
```

> **Interpretación esperada:** patrón aleatorio alrededor de 0 → **ok** (linealidad). Tendencias curvas sugieren especificación incorrecta o necesidad de transformaciones.

---

## 6.3 Homocedasticidad: Breusch–Pagan y White

```python
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

bp_test = het_breuschpagan(resid, model.model.exog)
white_test = het_white(resid, model.model.exog)

bp_labels = ["LM stat", "LM p-value", "F stat", "F p-value"]
white_labels = ["LM stat", "LM p-value", "F stat", "F p-value"]

bp_res = dict(zip(bp_labels, bp_test))
white_res = dict(zip(white_labels, white_test))

pd.DataFrame({"Breusch-Pagan": bp_res, "White": white_res})
```

> **Decisión:** p-value > 0.05 → **no se rechaza** homocedasticidad (ok).

---

## 6.4 Normalidad de residuos: QQ-plot y Shapiro–Wilk

```python
from scipy import stats

# QQ-plot
sm.qqplot(resid, line='45')
plt.title('QQ-plot de residuos')
plt.show()

# Shapiro (cuidado: sensible con n grande; usar como orientación)
W, p_shapiro = stats.shapiro(resid[:5000])  # limitar tamaño por restricción de la prueba
pd.DataFrame({"Shapiro-Wilk": {"W": W, "p-value": p_shapiro}})
```

> **Decisión:** puntos ~ línea (QQ-plot) y p-value Shapiro > 0.05 → **ok** (aprox. normalidad).

---

## 6.5 Autocorrelación: Durbin–Watson

```python
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(resid)
pd.DataFrame({"Durbin-Watson": {"stat": dw}})
```

> **Regla práctica:** valores cercanos a **2** sugieren ausencia de autocorrelación (ok).

---

## 6.6 Multicolinealidad: VIF

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_noint = data[sel].values  # sin intercepto
vif = [variance_inflation_factor(X_noint, i) for i in range(X_noint.shape[1])]

vif_tbl = pd.DataFrame({"variable": sel, "VIF": vif}).sort_values("VIF", ascending=False).reset_index(drop=True)
vif_tbl
```

> **Criterio:** VIF < 5 → aceptable; VIF 5–10 → revisar; VIF > 10 → fuerte colinealidad.

---

## 6.7 Resumen y decisiones

```python
# Compactamos resultados clave
summary = {}

# Homocedasticidad
summary["BP p-value"] = bp_res["LM p-value"]
summary["White p-value"] = white_res["LM p-value"]

# Normalidad
summary["Shapiro p-value"] = p_shapiro

# Autocorrelación
summary["Durbin-Watson"] = dw

# VIF (máximo y mediana)
summary["VIF max"] = float(np.max(vif))
summary["VIF median"] = float(np.median(vif))

# Tabla final
pd.DataFrame(summary, index=["resultado"]).T
```

> **Interpretación:**  
> - **Linealidad:** observar gráfico Residual vs Fitted (aleatorio → ok).  
> - **Homocedasticidad:** p > 0.05 en BP/White → ok.  
> - **Normalidad:** QQ-plot alineado y p Shapiro > 0.05 → ok (orientativo).  
> - **Autocorrelación:** Durbin–Watson ~ 2 → ok.  
> - **Multicolinealidad:** VIF máx. < 5 (ideal) o < 10 (tolerable).

---

## 6.8 Key takeaways

- El diagnóstico de supuestos valida la pertinencia del modelo OLS y guía ajustes (transformaciones, selección de variables, métodos robustos).  
- Documentar decisiones (qué se acepta/rechaza) permite trazabilidad y reproducibilidad del análisis.
