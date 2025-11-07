# Capítulo 8 · Selección de modelo y validación cruzada

> **Objetivo:** comparar especificaciones de modelos mediante métricas de ajuste (AIC, BIC, R² ajustado) y validar la capacidad predictiva con **K-Fold Cross Validation (CV)**, manteniendo el conjunto de variables candidatas como base.

---

## 8.1 Datos y modelo base

```python
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

# Cargar dataset (misma estructura que capítulos anteriores)
CANDIDATE_PATHS = [Path("data/ames_housing.csv"), Path("AmesHousing.csv")]
for p in CANDIDATE_PATHS:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("No se encontró data/ames_housing.csv ni AmesHousing.csv")

df = pd.read_csv(DATA_PATH)
target = "SalePrice"

# Selección automática de variables candidatas
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

ols = sm.OLS(y, X).fit()

sel, X.shape, ols.rsquared_adj, ols.aic, ols.bic
```
---

## 8.2 Comparación de especificaciones (reducción de variables)

Creamos versiones reducidas del modelo removiendo variables con **p-value > 0.05** o alta colinealidad (VIF alto).

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calcular VIF
X_noint = data[sel].values
vif = [variance_inflation_factor(X_noint, i) for i in range(X_noint.shape[1])]
vif_tbl = pd.DataFrame({"Variable": sel, "VIF": vif}).sort_values("VIF", ascending=False)
vif_tbl.head()

# Modelo reducido (ejemplo: eliminar variables con VIF > 10 o p>0.05)
pvals = ols.pvalues.drop("const", errors="ignore")
reduced_sel = [v for v in sel if pvals.get(v, 0) < 0.05]
if len(reduced_sel) < 3:
    reduced_sel = sel[:5]  # fallback

Xr = sm.add_constant(data[reduced_sel].values, has_constant="add")
ols_reduced = sm.OLS(y, Xr).fit()

pd.DataFrame({
    "Modelo": ["Completo", "Reducido"],
    "Variables": [len(sel), len(reduced_sel)],
    "R2_adj": [ols.rsquared_adj, ols_reduced.rsquared_adj],
    "AIC": [ols.aic, ols_reduced.aic],
    "BIC": [ols.bic, ols_reduced.bic]
})
```

**Interpretación:** modelos con **menor AIC/BIC** y **mayor R² ajustado** equilibran ajuste y parsimonia.

---

## 8.3 Validación cruzada (K-Fold CV)

```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

kf = KFold(n_splits=10, shuffle=True, random_state=123)

def cross_val_rmse(X, y):
    rmses = []
    for train, test in kf.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        model = sm.OLS(y_train, X_train).fit()
        y_pred = model.predict(X_test)
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    return np.mean(rmses), np.std(rmses)

# Comparación OLS completo vs reducido
X_full = sm.add_constant(data[sel].values, has_constant="add")
X_red  = sm.add_constant(data[reduced_sel].values, has_constant="add")

rmse_full_mean, rmse_full_std = cross_val_rmse(X_full, y)
rmse_red_mean, rmse_red_std   = cross_val_rmse(X_red, y)

pd.DataFrame({
    "Modelo": ["Completo", "Reducido"],
    "RMSE_mean": [rmse_full_mean, rmse_red_mean],
    "RMSE_std": [rmse_full_std, rmse_red_std]
})
```

**Interpretación:** un RMSE promedio menor indica **mejor capacidad predictiva** fuera de muestra.

---

## 8.4 Selección de modelo final

1. Si el modelo reducido mantiene R² similar y reduce AIC/BIC → **preferirlo** (más parsimonioso).  
2. Validar que no pierda capacidad predictiva (RMSE CV ≈ OLS completo).  
3. Revisar estabilidad de coeficientes (no deben variar drásticamente).  

---

## 8.5 Key takeaways

- La **selección de modelo** busca equilibrio entre ajuste (R², AIC/BIC) y generalización (CV).  
- **Validación cruzada** estima error fuera de muestra y evita sobreajuste.  
- **Modelos más simples** suelen ser preferibles si el desempeño predictivo se mantiene.  
- Esta fase prepara la base para aplicar **regularización (Ridge/Lasso)** o métodos predictivos más avanzados.
