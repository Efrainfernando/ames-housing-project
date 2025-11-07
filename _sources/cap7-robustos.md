# Capítulo 7 · Remedios y métodos robustos (research)

> **Objetivo:** evaluar la inferencia bajo **violaciones de supuestos** mediante (i) errores estándar robustos a heterocedasticidad (HC0–HC3), (ii) **regresión robusta** (RLM: Huber/Tukey) y (iii) **bootstrap** de coeficientes; y comparar con OLS clásico.

---

## 7.1 Datos y modelo base (mismas variables candidatas de cap. 4–6)

```python
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm

# Localiza el CSV (misma lógica que capítulos anteriores)
CANDIDATE_PATHS = [Path("data/ames_housing.csv"), Path("AmesHousing.csv")]
for p in CANDIDATE_PATHS:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("No se encontró data/ames_housing.csv ni AmesHousing.csv")

df = pd.read_csv(DATA_PATH)
target = "SalePrice"

# Selección automática de variables candidatas con bajo solapamiento (colinealidad)
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

# Ajuste OLS base
ols = sm.OLS(y, X).fit()

sel, X.shape, ols.rsquared
```
---

## 7.2 HC0–HC3: Errores estándar robustos a heterocedasticidad

```python
# Tipos disponibles: 'HC0', 'HC1', 'HC2', 'HC3'
hc0 = ols.get_robustcov_results(cov_type="HC0")
hc1 = ols.get_robustcov_results(cov_type="HC1")
hc2 = ols.get_robustcov_results(cov_type="HC2")
hc3 = ols.get_robustcov_results(cov_type="HC3")

# Extraer SE robustos e IC al 95%
def coef_se_ci(model_res, alpha=0.05):
    ci = model_res.conf_int(alpha=alpha)
    ci.columns = ["CI_low", "CI_high"]
    df = pd.DataFrame({
        "coef": model_res.params,
        "std_err": model_res.bse,
        "t": model_res.tvalues,
        "p>|t|": model_res.pvalues
    })
    return pd.concat([df, ci], axis=1)

ols_tbl  = coef_se_ci(ols)
hc0_tbl  = coef_se_ci(hc0)
hc1_tbl  = coef_se_ci(hc1)
hc2_tbl  = coef_se_ci(hc2)
hc3_tbl  = coef_se_ci(hc3)

# Comparación de amplitud de IC
cmp_ic = pd.DataFrame({
    "IC_width_OLS":  ols_tbl["CI_high"] - ols_tbl["CI_low"],
    "IC_width_HC0":  hc0_tbl["CI_high"] - hc0_tbl["CI_low"],
    "IC_width_HC1":  hc1_tbl["CI_high"] - hc1_tbl["CI_low"],
    "IC_width_HC2":  hc2_tbl["CI_high"] - hc2_tbl["CI_low"],
    "IC_width_HC3":  hc3_tbl["CI_high"] - hc3_tbl["CI_low"],
})

ols_tbl.head(), hc3_tbl.head(), cmp_ic.describe()
```
**Interpretación teórica breve:**  
- Los estimadores **HC** corrigen la subestimación de varianzas bajo heterocedasticidad.  
- **HC3** (aprox. jackknife) tiende a ser más conservador, recomendado en muestras pequeñas o con alto leverage.  
- La elección del estimador afecta **p-valores** y decisiones de significancia.

---

## 7.3 RLM: Regresiones robustas (Huber y Tukey)

```python
from statsmodels.robust.norms import HuberT, TukeyBiweight
from statsmodels.robust.robust_linear_model import RLM

# Ajustes robustos (misma X e y)
rlm_huber = RLM(y, X, M=HuberT()).fit()
rlm_tukey = RLM(y, X, M=TukeyBiweight()).fit()

# Tabla de coeficientes y pesos (para inspeccionar influencia)
def rlm_table(res):
    return pd.DataFrame({
        "coef": res.params,
        "std_err": res.bse,  # SE asintóticos RLM
    })

tbl_huber = rlm_table(rlm_huber)
tbl_tukey = rlm_table(rlm_tukey)

tbl_huber.head(), tbl_tukey.head()
```
**Características de las pérdidas:**  
- **Huber:** menos sensible a outliers moderados, buena eficiencia si los supuestos casi se cumplen.  
- **Tukey:** reduce fuertemente la influencia de outliers extremos (más robusta, menos eficiente).

---

## 7.4 Bootstrap de coeficientes (percentil, 95%)

```python
from sklearn.utils import resample

B = 1000  # ajusta según tiempo disponible
coefs = np.empty((B, X.shape[1]))
n = X.shape[0]
rng = np.random.default_rng(123)

for b in range(B):
    idx = rng.integers(0, n, n)  # remuestreo con reemplazo
    Xb = X[idx, :]
    yb = y[idx]
    coefs[b, :] = sm.OLS(yb, Xb).fit().params

coef_names = ["const"] + list(sel)
coef_boot = pd.DataFrame(coefs, columns=coef_names)

# Error estándar y IC percentilados (2.5%, 97.5%)
boot_se = coef_boot.std(ddof=1)
boot_ci = coef_boot.quantile([0.025, 0.975]).T
boot_ci.columns = ["CI_low_boot", "CI_high_boot"]

boot_summary = pd.concat([boot_se.rename("boot_se"), boot_ci], axis=1)
boot_summary.head()
```
**Ventajas del bootstrap:**  
- Sin supuestos distribucionales estrictos.  
- Captura asimetrías en distribuciones muestrales.  
- IC percentilados útiles cuando la normalidad es dudosa.

---

## 7.5 Tabla comparativa final (OLS vs HC3 vs Bootstrap)

```python
# Unificamos coeficientes, SE e IC para comparación resumida
final = pd.DataFrame({
    "coef_OLS": ols_tbl["coef"],
    "se_OLS":   ols_tbl["std_err"],
    "se_HC3":   hc3_tbl["std_err"],
})

# Amplitud IC 95% OLS y HC3
final["ICw_OLS"] = (ols_tbl["CI_high"] - ols_tbl["CI_low"])
final["ICw_HC3"] = (hc3_tbl["CI_high"] - hc3_tbl["CI_low"])

# Bootstrap (algunas filas pueden no coincidir en índice; reindex)
final = final.join(boot_summary, how="left")

# Ordenar por importancia aproximada (|t| de OLS)
t_abs = ols.tvalues
final = final.reindex(index=t_abs.abs().sort_values(ascending=False).index)

# Formateo bonito
display_cols = ["coef_OLS", "se_OLS", "se_HC3", "ICw_OLS", "ICw_HC3", "boot_se", "CI_low_boot", "CI_high_boot"]
final_round = final[display_cols].round(4)
final_round
```
**Lectura de la tabla:**  
- Compare **se_OLS** vs **se_HC3** vs **boot_se** (robustez).  
- Revise la **amplitud de IC**: **ICw_HC3** suele ser ≥ **ICw_OLS**; bootstrap puede ser asimétrico (ver extremos).  
- Diferencias grandes sugieren sensibilidad a heterocedasticidad o outliers → preferir HC3/RLM/transformaciones.

---

## 7.6 Key takeaways

- **HC0–HC3** corrigen varianzas bajo heterocedasticidad; **HC3** es más conservador (útil en n mediano/pequeño y alto leverage).  
- **RLM (Huber/Tukey)** reduce la influencia de outliers; útil cuando el OLS es inestable.  
- **Bootstrap** entrega SE e IC sin normalidad; ayuda a validar estabilidad de conclusiones.  
- La **tabla comparativa** orienta qué método comunica mejor la incertidumbre en tu aplicación concreta.
