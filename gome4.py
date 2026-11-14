from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def best_subset(x, y, names):
    answers = []
    for k in range(1, len(names) + 1):
        best_mse = float("inf")
        best_names = None
        for group in combinations(names, k):
            model = LinearRegression().fit(x[list(group)], y)
            mse = mean_squared_error(y, model.predict(x[list(group)]))
            if mse < best_mse:
                best_mse = mse
                best_names = group
        answers.append((k, best_mse, best_names))
        print(k, round(best_mse, 2), best_names)
    return answers


hitters = pd.read_csv("Hitters.csv").dropna()
hitters = pd.get_dummies(
    hitters,
    columns=["League", "Division", "NewLeague"],
    drop_first=True
)

x = hitters.drop(columns="Salary")
y = hitters["Salary"]
print("Hitters observations:", len(y))
print("Best subset selection")
best_subset(x.iloc[:, :10], y, list(x.columns[:10]))

college = pd.read_csv("College.csv")
college = college.rename(columns={college.columns[0]: "CollegeName"})
college["Private"] = (college["Private"] == "Yes").astype(int)

y = college["Room.Board"]
five = ["Accept", "Enroll", "Outstate", "Books", "Grad.Rate"]

pd.plotting.scatter_matrix(college[five + ["Room.Board"]], figsize=(10, 10))
plt.tight_layout()
plt.savefig("question4_pairplot.png")
plt.close()

print("\nCorrelations")
print(college[five + ["Room.Board"]].corr()["Room.Board"])

print("\nCollege best subset selection")
best_model = None
for k in range(0, 6):
    for group in combinations(five, k):
        if k == 0:
            adjusted_r2 = 0
        else:
            model = LinearRegression().fit(college[list(group)], y)
            r2 = model.score(college[list(group)], y)
            n = len(y)
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        print(k, group, round(adjusted_r2, 5))
        if best_model is None or adjusted_r2 > best_model[0]:
            best_model = (adjusted_r2, group)
print("Selected:", best_model)

predictors = [c for c in college.columns if c not in ["CollegeName", "Room.Board"]]
x = college[predictors]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

ols = LinearRegression().fit(x_train, y_train)
print("\nOLS MSPE:", mean_squared_error(y_test, ols.predict(x_test)))

scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)
lambdas = [10 ** (-4 + i * 10 / 99) for i in range(100)]

ridge_errors = []
for value in lambdas:
    model = Ridge(alpha=value).fit(x_train, y_train)
    ridge_errors.append(mean_squared_error(y_test, model.predict(x_test)))
ridge_i = min(range(len(ridge_errors)), key=ridge_errors.__getitem__)
print("Ridge lambda:", lambdas[ridge_i])
print("Ridge MSPE:", ridge_errors[ridge_i])

plt.plot(lambdas, ridge_errors)
plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("MSPE")
plt.savefig("question4_ridge_mspe.png")
plt.close()

lasso_errors = []
for value in lambdas:
    model = Lasso(alpha=value, max_iter=100000).fit(x_train, y_train)
    lasso_errors.append(mean_squared_error(y_test, model.predict(x_test)))
lasso_i = min(range(len(lasso_errors)), key=lasso_errors.__getitem__)
print("Lasso lambda:", lambdas[lasso_i])
print("Lasso MSPE:", lasso_errors[lasso_i])

plt.plot(lambdas, lasso_errors)
plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("MSPE")
plt.savefig("question4_lasso_mspe.png")
plt.close()
