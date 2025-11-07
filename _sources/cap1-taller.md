# Capítulo 1 · Demostraciones solicitadas

## 1.1 Modelo y objetivo

Sea el modelo de **regresión lineal simple**
\[
y_i=\beta_0+\beta_1 x_i+\varepsilon_i,\quad i=1,\ldots,n,
\]
con \(\mathbb E(\varepsilon_i)=0\) y \(\operatorname{Var}(\varepsilon_i)=\sigma^2\), mutuamente independientes.

Demuestre que la suma de cuadrados de los residuos dividida por \(\sigma^2\),
\[
\frac{SS_{\text{Res}}}{\sigma^2}=\frac{\sum_{i=1}^n e_i^2}{\sigma^2},
\]
es una **combinación cuadrática** de los errores y que
\[
\frac{SS_{\text{Res}}}{\sigma^2}\sim\chi^2_{\,n-2}.
\]
Explique por qué se **restan dos grados de libertad** en el modelo simple (asociados a \(\hat\beta_0\) y \(\hat\beta_1\)).

---

## 1.2 Demostración (vía álgebra matricial)

Sea \(X=[\mathbf 1,\; x]\in\mathbb R^{n\times 2}\), \(H=X(X'X)^{-1}X'\) la **matriz sombrero** y \(M=I-H\).
Los residuos cumplen \(e=My\) y, dado que \(y=X\beta+\varepsilon\),
\[
e=M(X\beta+\varepsilon)=M\varepsilon \quad\Rightarrow\quad
SS_{\text{Res}}=e'e=\varepsilon' M\, \varepsilon.
\]
Propiedades: \(M\) es **simétrica** e **idempotente** (\(M^2=M\)) y \(\operatorname{rango}(M)=n-\operatorname{rango}(H)=n-2\).
Como \(\varepsilon\sim \mathcal N(0,\sigma^2 I)\), entonces
\[
\frac{\varepsilon' M\, \varepsilon}{\sigma^2}\sim \chi^2_{\,\operatorname{rango}(M)}=\chi^2_{\,n-2}.
\]
Por lo tanto, \(\dfrac{SS_{\text{Res}}}{\sigma^2}\sim\chi^2_{\,n-2}\).

**Interpretación.** Los **2 grados de libertad** se consumen al estimar \(\beta_0\) y \(\beta_1\).

---

## 1.3 Observaciones prácticas

- En el modelo con \(p\) parámetros (incluye intercepto), \(\dfrac{SS_{\text{Res}}}{\sigma^2}\sim\chi^2_{\,n-p}\).
- Un estimador insesgado de \(\sigma^2\) es \(s^2=SS_{\text{Res}}/(n-2)\).

---

## 1.4 Referencia

Draper, N. R., & Smith, H. (1998). *Applied Regression Analysis*. Wiley.