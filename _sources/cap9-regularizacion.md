# Capítulo 9 · Regularización: Ridge y Lasso

> **Objetivo:** aplicar técnicas de **regularización (Ridge y Lasso)** para mitigar la multicolinealidad y mejorar la estabilidad de los estimadores, comparando su desempeño con el modelo OLS clásico.

---

## 9.1 Introducción teórica

La **regularización** introduce una penalización sobre el tamaño de los coeficientes para evitar sobreajuste.  
Sea el modelo lineal clásico \( y = X\beta + \varepsilon \). Los estimadores se definen como:

**Ridge (L2):**  
  
$$ \hat{\beta}_{ridge} = \arg\min_\beta \left\{ \|y - X\beta\|^2 + \lambda \|\beta\|_2^2 \right\} $$  
  
**Lasso (L1):**  
  
$$ \hat{\beta}_{lasso} = \arg\min_\beta \left\{ \|y - X\beta\|^2 + \lambda \|\beta\|_1 \right\} $$  
  
- Ridge reduce la varianza de los coeficientes correlacionados pero **no los lleva a cero**.  
- Lasso puede **anular completamente** algunos coeficientes, realizando selección automática de variables.

---

## 9.2 Preparación de datos

Reutilizamos el conjunto de **variables candidatas** de capítulos previos y estandarizamos las covariables antes de aplicar la penalización.

```python
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

# Cargar dataset (misma estructura)
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
X = data[sel].values

sel, X.shape
```

---

## 9.3 Implementación de RidgeCV y LassoCV

Usamos **validación cruzada (10-Fold)** para determinar el valor óptimo de \( \lambda \) (alpha).

```python
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

kf = KFold(n_splits=10, shuffle=True, random_state=42)
alphas = np.logspace(-3, 3, 50)

# Ridge con validación cruzada automática
ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=kf))
ridge.fit(X, y)
ridge_alpha = ridge.named_steps["ridgecv"].alpha_

# Lasso con validación cruzada automática
lasso = make_pipeline(StandardScaler(), LassoCV(alphas=alphas, cv=kf, random_state=42, max_iter=10000))
lasso.fit(X, y)
lasso_alpha = lasso.named_steps["lassocv"].alpha_

ridge_alpha, lasso_alpha
```

---

## 9.4 Comparación de coeficientes y desempeño

```python
# Ajuste OLS clásico para comparar
X_sm = sm.add_constant(X, has_constant="add")
ols = sm.OLS(y, X_sm).fit()

ridge_preds = ridge.predict(X)
lasso_preds = lasso.predict(X)

ridge_r2 = r2_score(y, ridge_preds)
lasso_r2 = r2_score(y, lasso_preds)

coef_df = pd.DataFrame({
    "Variable": sel,
    "OLS_coef": ols.params[1:len(sel)+1],
    "Ridge_coef": ridge.named_steps["ridgecv"].coef_,
    "Lasso_coef": lasso.named_steps["lassocv"].coef_
}).round(4)

coef_df["Lasso_Zero?"] = coef_df["Lasso_coef"].apply(lambda v: abs(v) < 1e-6)
coef_df
```

> **Interpretación:** Los coeficientes Ridge son una versión suavizada de OLS.  
> En Lasso, varios coeficientes pueden ser exactamente cero, indicando selección automática de variables.

---

## 9.5 Evaluación de error y métricas

```python
ridge_rmse = np.sqrt(mean_squared_error(y, ridge_preds))
lasso_rmse = np.sqrt(mean_squared_error(y, lasso_preds))
ols_rmse = np.sqrt(mean_squared_error(y, ols.predict(X_sm)))

pd.DataFrame({
    "Modelo": ["OLS", "Ridge", "Lasso"],
    "R²": [ols.rsquared, ridge_r2, lasso_r2],
    "RMSE": [ols_rmse, ridge_rmse, lasso_rmse],
    "Alpha óptimo": ["-", ridge_alpha, lasso_alpha]
})
```

> **Conclusión:** los modelos penalizados pueden mejorar el error cuadrático medio (RMSE) y reducir la varianza de los coeficientes, especialmente en presencia de multicolinealidad.

---

## 9.6 Visualización de shrinkage

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.scatter(coef_df["OLS_coef"], coef_df["Ridge_coef"], label="Ridge", alpha=0.7)
plt.scatter(coef_df["OLS_coef"], coef_df["Lasso_coef"], label="Lasso", alpha=0.7)
plt.axhline(0, color="gray", linestyle="--")
plt.axvline(0, color="gray", linestyle="--")
plt.xlabel("OLS Coefficients")
plt.ylabel("Regularized Coefficients")
plt.title("Comparación de shrinkage: Ridge y Lasso")
plt.legend()
plt.show()
```

> El gráfico ilustra cómo Ridge reduce los coeficientes sin anularlos y cómo Lasso los **contrae hacia cero** (algunos completamente).

---

## 9.7 Key takeaways

- **Ridge (L2)** mejora la estabilidad del modelo ante predictores correlacionados.  
- **Lasso (L1)** realiza selección automática de variables, mejorando la interpretabilidad.  
- Ambos reducen la varianza de los estimadores y mitigan el sobreajuste.  
- En aplicaciones reales, puede combinarse ambos enfoques en un modelo **ElasticNet**.
