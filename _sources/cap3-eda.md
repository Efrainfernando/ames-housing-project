# Capítulo 3 · Análisis exploratorio (EDA)

> **Objetivo:** detectar relaciones y elegir variables relevantes para modelar `SalePrice` en *Ames Housing*.
>
> Este capítulo genera gráficos y resúmenes automáticos: histogramas/boxplots de `SalePrice`, dispersión con `GrLivArea`, heatmap de correlaciones y comparaciones por categorías (`Neighborhood`, `OverallQual`). Al final propone una lista de **variables candidatas** (8–12) con justificación básica.

---

## 3.1 Carga del dataset y preparación mínima

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ruta estándar del proyecto
DATA_PATHS = [Path('data/ames_housing.csv'), Path('AmesHousing.csv')]
for p in DATA_PATHS:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError('No se encontró data/ames_housing.csv ni AmesHousing.csv')

df = pd.read_csv(DATA_PATH)

# Variables de interés inicial
target = 'SalePrice'
basic_feats = ['GrLivArea', 'OverallQual', 'TotalBsmtSF', 'GarageCars', 'GarageArea',
               'YearBuilt', 'FullBath', 'LotArea']

# Asegurar tipos
if df[target].dtype == 'O':
    df[target] = pd.to_numeric(df[target], errors='coerce')

# Vista general
print(df.shape)
df.head(3)
```
---

## 3.2 Histograma y boxplot de `SalePrice`

```python
col = 'SalePrice'
series = df[col].dropna()

# Histograma (escala lineal)
plt.figure(figsize=(6,4))
plt.hist(series, bins=40)
plt.title('Histograma de SalePrice')
plt.xlabel('SalePrice')
plt.ylabel('Frecuencia')
plt.show()

# Histograma (log10) para ver simetría
plt.figure(figsize=(6,4))
plt.hist(np.log10(series), bins=40)
plt.title('Histograma de log10(SalePrice)')
plt.xlabel('log10(SalePrice)')
plt.ylabel('Frecuencia')
plt.show()

# Boxplot simple
plt.figure(figsize=(4,4))
plt.boxplot(series.dropna(), vert=True, labels=['SalePrice'])
plt.title('Boxplot de SalePrice')
plt.ylabel('Valor')
plt.show()
```
---

## 3.3 Dispersión: `SalePrice` vs `GrLivArea`

```python
x, y = df['GrLivArea'], df['SalePrice']

mask = x.notna() & y.notna()
x, y = x[mask], y[mask]

plt.figure(figsize=(6,4))
plt.scatter(x, y, s=12, alpha=0.6)
plt.title('SalePrice vs GrLivArea')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()

# Versión log del precio para linealidad
plt.figure(figsize=(6,4))
plt.scatter(x, np.log1p(y), s=12, alpha=0.6)
plt.title('log(1+SalePrice) vs GrLivArea')
plt.xlabel('GrLivArea')
plt.ylabel('log(1+SalePrice)')
plt.show()
```
---

## 3.4 Heatmap de correlaciones principales

Seleccionamos las **12** variables numéricas con mayor correlación (absoluta) con `SalePrice` y mostramos su matriz de correlación.

```python
# Seleccionar numéricas
num_df = df.select_dtypes(include=[np.number])
num_df = num_df.replace([np.inf, -np.inf], np.nan)

# Top por correlación absoluta con SalePrice
corr_target = num_df.corr(numeric_only=True)[target].dropna().abs().sort_values(ascending=False)
top_feats = corr_target.index[1:13]  # excluir SalePrice mismo (posición 0), tomar 12 siguientes
print('Top correlaciones con SalePrice:\n', corr_target.head(13))

corr_mat = num_df[[target, *top_feats]].corr()

# Heatmap manual con matplotlib
fig, ax = plt.subplots(figsize=(7,6))
im = ax.imshow(corr_mat, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_mat.columns)))
ax.set_yticks(range(len(corr_mat.index)))
ax.set_xticklabels(corr_mat.columns, rotation=45, ha='right')
ax.set_yticklabels(corr_mat.index)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.title('Matriz de correlación (top con SalePrice)')
plt.tight_layout()
plt.show()
```
---

## 3.5 Boxplots por categorías

### 3.5.1 `OverallQual` (ordinal)

```python
col = 'OverallQual'
tmp = df[[col, target]].dropna()
groups = [g[target].values for _, g in tmp.groupby(col)]
labels = [str(int(v)) for v in sorted(tmp[col].unique())]

plt.figure(figsize=(7,4))
plt.boxplot(groups, labels=labels, showfliers=False)
plt.title(f'SalePrice por {col}')
plt.xlabel(col)
plt.ylabel('SalePrice')
plt.show()
```

### 3.5.2 `Neighborhood` (muchas categorías)

Mostramos **las 10 más frecuentes** (mediana del precio por barrio) para evitar saturación visual.

```python
col = 'Neighborhood'
topN = 10
freq = df[col].value_counts().head(topN).index
med = (df[df[col].isin(freq)]
       .groupby(col)[target].median()
       .sort_values(ascending=False))

plt.figure(figsize=(7,4))
med.plot(kind='bar')
plt.title('Mediana de SalePrice por Neighborhood (Top 10)')
plt.xlabel('Neighborhood')
plt.ylabel('Mediana de SalePrice')
plt.tight_layout()
plt.show()
```
---

## 3.6 Selección de variables candidatas (8–12)

Criterios usados:
1) Correlación absoluta alta con `SalePrice` (top k).  
2) Evitar **multicolinealidad**: si dos variables tienen |ρ|>0.85, quedarse con una (la más interpretable).  
3) Preferir variables con significado estructural (calidad, área, antigüedad).

```python
k = 15  # empezamos con un pool un poco mayor
corr_abs = num_df.corr(numeric_only=True)[target].dropna().abs().sort_values(ascending=False)
pool = list(corr_abs.index[1:k+1])  # quitar SalePrice

# Filtrado por colinealidad (>0.85)
sel = []
for v in pool:
    if not sel:
        sel.append(v)
        continue
    ok = True
    for u in sel:
        r = abs(num_df[[v, u]].corr().iloc[0,1])
        if r > 0.85:
            ok = False
            break
    if ok:
        sel.append(v)
    if len(sel) >= 12:
        break

print('Variables candidatas (<=12, sin alta colinealidad):')
for i, v in enumerate(sel, 1):
    print(f'{i:>2}. {v}  (|ρ|={corr_abs[v]:.3f})')
```

> **Sugerencia:** considera transformar a log `SalePrice`, `GrLivArea`, `LotArea`, y/o `TotalBsmtSF` si muestran asimetría fuerte y outliers.  
> Las categorías ordinales (`OverallQual`, `KitchenQual`, etc.) pueden utilizarse tal cual o recodificarse según su escala.

---

## 3.7 Key takeaways

- `SalePrice` es asimétrica; una transformación log mejora linealidad y homocedasticidad.  
- `GrLivArea`, `OverallQual`, `GarageCars/Area`, `TotalBsmtSF`, `YearBuilt` suelen correlacionarse fuertemente con el precio.  
- Evitar variables casi duplicadas (p. ej., `GarageArea` vs `GarageCars`).  
- La lista de candidatas prioriza interpretación y baja colinealidad para el modelo base del capítulo OLS.
