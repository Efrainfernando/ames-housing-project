# Capítulo 10 · ElasticNet y comparación final de modelos

> **Objetivo:** aplicar **ElasticNet** y comparar su desempeño con OLS, Ridge y Lasso, usando las mismas **variables candidatas** de capítulos previos.  

---

## 10.1 Introducción teórica

ElasticNet combina las penalizaciones **L1 (Lasso)** y **L2 (Ridge)** en un único problema de optimización:

  
$$ \hat{\beta}_{EN} = \arg\min_\beta \Big\{ \|y - X\beta\|^2 + \lambda \big[ \alpha \|\beta\|_1 + (1-\alpha)\|\beta\|_2^2 \big] \Big\} $$  
  

- \( \alpha \in [0,1] \): pondera entre Lasso (1) y Ridge (0).  
- \( \lambda > 0 \): controla la intensidad de la regularización.  

---

## 10.2 Datos y construcción de variables (misma lógica que caps. 4–9)

```python
import numpy as np
import pandas as pd
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

# Selección de variables candidatas con control de colinealidad
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
X = data[sel].values

# Ajuste OLS para comparación
X_sm = sm.add_constant(X, has_constant="add")
ols = sm.OLS(y, X_sm).fit()

sel, X.shape, ols.rsquared_adj
```
---

## 10.3 Modelos Ridge, Lasso y ElasticNet con validación cruzada

```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

kf = KFold(n_splits=10, shuffle=True, random_state=42)
alphas = np.logspace(-3, 3, 50)
l1_ratios = np.linspace(0.1, 0.9, 9)

# Ridge
ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=kf))
ridge.fit(X, y)
ridge_alpha = ridge.named_steps["ridgecv"].alpha_

# Lasso
lasso = make_pipeline(StandardScaler(), LassoCV(alphas=alphas, cv=kf, random_state=42, max_iter=10000))
lasso.fit(X, y)
lasso_alpha = lasso.named_steps["lassocv"].alpha_

# ElasticNet
elastic = make_pipeline(
    StandardScaler(),
    ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=kf, random_state=42, max_iter=10000)
)
elastic.fit(X, y)
elastic_alpha = elastic.named_steps["elasticnetcv"].alpha_
elastic_l1ratio = elastic.named_steps["elasticnetcv"].l1_ratio_

ridge_alpha, lasso_alpha, (elastic_alpha, float(elastic_l1ratio))
```
---

## 10.4 Comparación de coeficientes

```python
elastic_coefs = elastic.named_steps["elasticnetcv"].coef_

coef_comp = pd.DataFrame({
    "Variable": sel,
    "OLS": ols.params[1:len(sel)+1],
    "Ridge": ridge.named_steps["ridgecv"].coef_,
    "Lasso": lasso.named_steps["lassocv"].coef_
}).round(4)

coef_comp["ElasticNet"] = np.round(elastic_coefs, 4)
coef_comp["ElasticNet_Zero?"] = coef_comp["ElasticNet"].apply(lambda v: abs(v) < 1e-6)
coef_comp
```

> **Lectura:** ElasticNet combina la estabilidad de Ridge y la parsimonia de Lasso; algunos coeficientes pueden quedar exactamente en cero.

---

## 10.5 Evaluación comparativa (R² y RMSE in-sample)

```python
ridge_preds = ridge.predict(X)
lasso_preds = lasso.predict(X)
elastic_preds = elastic.predict(X)

from sklearn.metrics import mean_squared_error, r2_score
ridge_r2 = r2_score(y, ridge_preds)
lasso_r2 = r2_score(y, lasso_preds)
elastic_r2 = r2_score(y, elastic_preds)

ridge_rmse = np.sqrt(mean_squared_error(y, ridge_preds))
lasso_rmse = np.sqrt(mean_squared_error(y, lasso_preds))
elastic_rmse = np.sqrt(mean_squared_error(y, elastic_preds))
ols_rmse = np.sqrt(mean_squared_error(y, ols.predict(X_sm)))

comparison = pd.DataFrame({
    "Modelo": ["OLS", "Ridge", "Lasso", "ElasticNet"],
    "R²": [ols.rsquared, ridge_r2, lasso_r2, elastic_r2],
    "RMSE": [ols_rmse, ridge_rmse, lasso_rmse, elastic_rmse]
})
comparison
```

---

## 10.6 Gráficas de comparación

```python
import matplotlib.pyplot as plt

# RMSE
plt.figure(figsize=(7,4))
plt.bar(comparison["Modelo"], comparison["RMSE"])
plt.ylabel("RMSE")
plt.title("Comparación de error (RMSE)")
plt.show()

# R²
plt.figure(figsize=(7,4))
plt.bar(comparison["Modelo"], comparison["R²"])
plt.ylabel("R²")
plt.title("Comparación de desempeño (R²)")
plt.show()
```

> *Nota:* para una comparación más realista, usar **validación cruzada** fuera de muestra (ver Cap. 8).

---

## 10.7 Selección final y conclusiones

- Si **ElasticNet** logra **RMSE** bajo y **R²** alto con buena parsimonia, se recomienda como **modelo final**.  
- **Ridge** es preferible con muchos predictores correlacionados relevantes.  
- **Lasso** útil para identificar un subconjunto mínimo de variables.  
- Parámetros óptimos hallados:  
  - Ridge: \( \lambda^\* = \) `ridge_alpha`  
  - Lasso: \( \lambda^\* = \) `lasso_alpha`  
  - ElasticNet: \( \lambda^\* = \) `elastic_alpha`, \( \alpha^\* = \) `elastic_l1ratio`  

---

## 10.8 Key takeaways

- **ElasticNet** equilibra selección de variables y estabilidad numérica.  
- La **regularización** mitiga multicolinealidad y reduce varianza de los estimadores.  
- La comparación conjunta OLS–Ridge–Lasso–ElasticNet facilita una **decisión informada** del modelo final.
