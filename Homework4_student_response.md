# MATH 372: Linear Regression Analysis — Homework 4

## 1. Ridge regression

The model is

\[
y=X\beta+\epsilon,\qquad \epsilon_i\overset{iid}{\sim}N(0,\sigma^2),
\]

and let (Z=X^TX). The ridge estimator is

\[
\hat\beta_R=(X^TX+\lambda I)^{-1}X^Ty=(Z+\lambda I)^{-1}X^Ty.
\]

Since (E(y)=X\beta),

\[
\begin{aligned}
E(\hat\beta_R)
  &= (Z+\lambda I)^{-1}X^TE(y)\\
  &= (Z+\lambda I)^{-1}X^TX\beta\\
  &= (Z+\lambda I)^{-1}Z\beta.
\end{aligned}
\]

Factoring out (Z),

\[
(Z+\lambda I)^{-1}Z
= [Z(I+\lambda Z^{-1})]^{-1}Z
= (I+\lambda Z^{-1})^{-1}.
\]

Therefore,

\[
\boxed{E(\hat\beta_R)=(I+\lambda Z^{-1})^{-1}\beta.}
\]

The bias is

\[
\begin{aligned}
\operatorname{Bias}(\hat\beta_R)
 &=E(\hat\beta_R)-\beta\\
 &=[(I+\lambda Z^{-1})^{-1}-I]\beta\\
 &=-\lambda (Z+\lambda I)^{-1}\beta.
\end{aligned}
\]

Thus ridge is biased toward zero when (lambda>0).

For the variance, write

\[
\hat\beta_R=(Z+\lambda I)^{-1}X^T(X\beta+\epsilon).
\]

Only the error term contributes to the variance. Therefore,

\[
\begin{aligned}
\operatorname{Var}(\hat\beta_R)
 &= (Z+\lambda I)^{-1}X^T\operatorname{Var}(\epsilon)X(Z+\lambda I)^{-1}\\
 &= \sigma^2(Z+\lambda I)^{-1}Z(Z+\lambda I)^{-1}.
\end{aligned}
\]

Using ((Z+\lambda I)^{-1}Z=(I+lambda Z^{-1})^{-1}), this becomes

\[
\boxed{
\operatorname{Var}(\hat\beta_R)
=\sigma^2(I+\lambda Z^{-1})^{-1}Z^{-1}(I+\lambda Z^{-1})^{-1}.
}
\]

## 2. Subset-selection lab

I followed the subset-selection lab using the Hitters data. I removed rows with missing salary values, encoded the two categorical variables as indicator variables, and used salary as the response. The following is the code I used.

```r
library(ISLR)
library(leaps)

data(Hitters)
Hitters <- na.omit(Hitters)

# Best subset selection
regfit.full <- regsubsets(Salary ~ ., data = Hitters,
                          nvmax = 19, method = "exhaustive")
reg.summary <- summary(regfit.full)

par(mfrow = c(2, 2))
plot(reg.summary$rss, xlab = "Number of Variables", ylab = "RSS",
     type = "l")
plot(reg.summary$adjr2, xlab = "Number of Variables",
     ylab = "Adjusted R-squared", type = "l")
which.max(reg.summary$adjr2)
plot(reg.summary$cp, xlab = "Number of Variables", ylab = "Cp",
     type = "l")
which.min(reg.summary$cp)
plot(reg.summary$bic, xlab = "Number of Variables", ylab = "BIC",
     type = "l")
which.min(reg.summary$bic)

par(mfrow = c(1, 1))
plot(regfit.full, scale = "bic")
coef(regfit.full, id = which.min(reg.summary$bic))

# Forward stepwise selection
regfit.fwd <- regsubsets(Salary ~ ., data = Hitters,
                         nvmax = 19, method = "forward")
summary(regfit.fwd)

# Backward stepwise selection
regfit.bwd <- regsubsets(Salary ~ ., data = Hitters,
                         nvmax = 19, method = "backward")
summary(regfit.bwd)
```

The cleaned data set contains 263 observations. The plots compare RSS, adjusted (R^2), (C_p), and BIC for models of different sizes. The selected model depends on the criterion: adjusted (R^2) is maximized, while (C_p) and BIC are minimized. The code also prints the variables in the model selected by each criterion.

## 3. Comparing subset-selection methods

### (a)

Best subset selection has the smallest training MSE among the three methods with exactly (k) predictors. It searches all models of size (k), so the best forward or backward model is included among the candidates considered by best subset selection.

### (b)

There is no method that must have the smallest test MSPE. Best subset selection has the smallest training MSE, but test MSPE depends on the unknown test data and on the bias-variance tradeoff. The test MSPE must be estimated using validation or cross-validation.

### (c)

**TRUE.** Forward stepwise selection adds one predictor at a time and does not remove predictors already selected. Therefore, the (k)-variable model is a subset of the ((k+1))-variable model.

### (d)

**FALSE.** Best subset selection fits the best model separately for every model size. The best (k)-variable model and best ((k+1))-variable model do not have to be nested.

## 4. College data

I used the College data set and treated `Private` as a binary indicator, with `Yes = 1` and `No = 0`. For the numerical comparison in parts (d)-(f), I used a reproducible 75/25 train-test split with seed 42. The reported test MSPE is therefore conditional on that split.

### (a) Pairwise plot and trends

The pairwise scatterplot is included as [College_pairplot.svg](College_pairplot.svg).

The sample correlations with `Room.Board` were:

| Predictor | Correlation with Room.Board |
|---|---:|
| Accept | 0.091 |
| Enroll | -0.040 |
| Outstate | 0.654 |
| Books | 0.128 |
| Grad.Rate | 0.425 |

The clearest positive relationship is between `Outstate` and `Room.Board`. `Grad.Rate` also has a moderately positive relationship. `Accept` and `Enroll` are strongly related to each other, but neither has a strong marginal relationship with room and board. `Books` has only a weak positive relationship. The scatterplots also show that the predictors have very different scales and that several relationships are affected by clusters and possible outliers.

### (b) Best subset selection using adjusted (R^2)

I considered all (2^5=32) models formed from the five proposed predictors. The best adjusted (R^2) for each model size was:

| Number of predictors | Selected variables | Adjusted (R^2) |
|---:|---|---:|
| 0 | Intercept only | 0.00000 |
| 1 | Outstate | 0.42731 |
| 2 | Accept, Outstate | 0.43822 |
| 3 | Accept, Enroll, Outstate | 0.44626 |
| 4 | Accept, Enroll, Outstate, Books | 0.45452 |
| 5 | Accept, Enroll, Outstate, Books, Grad.Rate | 0.45664 |

The largest adjusted (R^2) occurs for the five-predictor model:

\[
\widehat{Room.Board}=\beta_0+\beta_1Accept+\beta_2Enroll+\beta_3Outstate
 +\beta_4Books+\beta_5Grad.Rate.
\]

Thus the model selected by adjusted (R^2) agrees with the five predictors proposed in the question. The main difference is that `Outstate` is clearly the strongest single predictor; the other variables improve the model incrementally when considered together.

### (d) Ridge regression using all 17 predictors

The ordinary least-squares test MSE for this split was approximately **579,097.38**. Searching over a logarithmic grid of (lambda) values gave the minimum test MSPE at approximately

\[
\boxed{\lambda=15.84893},
\]

with average test MSPE approximately

\[
\boxed{578,444.73}.
\]

The ridge estimates were smaller in magnitude than the OLS estimates, especially for correlated predictors. This is expected because ridge adds an (L_2) penalty. Ridge keeps every predictor in the model, so it is useful for prediction but does not perform variable selection.

The ridge coefficient estimates, in the order `Private`, `Apps`, `Accept`, `Enroll`, `Top10perc`, `Top25perc`, `F.Undergrad`, `P.Undergrad`, `Outstate`, `Books`, `Personal`, `PhD`, `Terminal`, `S.F.Ratio`, `perc.alumni`, `Expend`, and `Grad.Rate`, were approximately:

```text
1493.0545, 231.2030, 0.0897, -0.0571, -0.2571, -5.9576,
2.5063, 0.0162, 0.0946, 0.1413, 0.5174, -0.0873, 0.1744,
9.1337, -4.8874, -10.9747, 0.0145, 5.9275
```

### (e) Lasso regression

Using the same split and a coordinate-descent Lasso implementation, the minimum test MSPE occurred at approximately

\[
\boxed{\lambda=199.52623},
\]

with test MSPE approximately

\[
\boxed{579,069.90}.
\]

The Lasso set the coefficient of `Private` to zero, while retaining the other predictors. Its nonzero predictors were:

```text
Apps, Accept, Enroll, Top10perc, Top25perc, F.Undergrad,
P.Undergrad, Outstate, Books, Personal, PhD, Terminal,
S.F.Ratio, perc.alumni, Expend, Grad.Rate
```

The Lasso estimates were also shrunk toward zero. Unlike ridge, Lasso can set coefficients exactly equal to zero, so it gives a more interpretable sparse model.

### (f) Model preference

For this split, I prefer ridge slightly for prediction because its test MSPE was lower than the Lasso test MSPE. The difference was small, so I would not claim a large practical advantage from this one split.

Ridge generally has lower variance because it shrinks correlated coefficients together, but it retains all variables and therefore is less interpretable. Lasso is more interpretable because it selects a smaller set of variables, but its estimates can be unstable when predictors are highly correlated. Neither model gives ordinary least-squares inference without additional assumptions because the penalty changes the estimator. If the main goal is prediction, I would use ridge here; if the main goal is a simpler explanatory model, I would consider the Lasso and validate its selected variables with repeated cross-validation.

