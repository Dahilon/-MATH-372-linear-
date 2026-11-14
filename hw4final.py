import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

print("Homework 4")


# Question 1 is written out in the paper
# Question 3 is also written out in the paper


# Question 2
# This section loads the Hitters data and removes missing values.

hit = pd.read_csv("Hitters.csv")
hit = hit.dropna()
hit = pd.get_dummies(hit, columns=["League", "Division", "NewLeague"],
                     drop_first=True)
y = hit["Salary"]
X = hit.drop("Salary", axis=1)

print("number of rows", len(hit)),print("number of columns", len(X.columns))
def get_mse(cols):
    reg = LinearRegression()
    reg.fit(X[list(cols)], y)
    yh = reg.predict(X[list(cols)])
    return mean_squared_error(y, yh)
all_results = []

# I used 8 as the maax because checking every possible 19-variable model takes a long time
# Question 2 
max_number = 8
for num in range(1, max_number + 1):
    best = 999999999999
    best_cols = []
    for cols in combinations(X.columns, num):
        this_mse = get_mse(cols)
        if this_mse < best:
            best = this_mse
            best_cols = cols
            reg = LinearRegression().fit(X[list(best_cols)], y)
            r2 = reg.score(X[list(best_cols)], y)
            n = len(y)
            p = len(best_cols)
            adj = 1 - ((1-r2)*(n-1)/(n-p-1))
    bic = n*np.log(best) + p*np.log(n)
    all_results.append([num, best, adj, bic, best_cols])
    print("model with", num, "variables")
    print("mse =", round(best, 2))
    print("adjusted r2 =", round(adj, 4))
    print("bic =", round(bic, 2))
    print(best_cols)
    print()
    print("best adjusted r2 model"),print(max(all_results, key=lambda x: x[2])),print("best bic model"),print(min(all_results, key=lambda x: x[3]))

# Question 2 forward stepwise selection.
now = []
left = list(X.columns)
for z in range(max_number):
    bestone = None
    bestmse = 999999999999
    for a in left:
        trycols = now + [a]
        mse1 = get_mse(trycols)
        if mse1 < bestmse:
            bestmse = mse1
            bestone = a
            now.append(bestone)
    left.remove(bestone)
    print("forward", z+1, now)
# Question 2 backward stepwise selection.
now2 = list(X.columns)
while len(now2) > max_number:
    removeone = None
    smallest_mse = 999999999999
    for a in now2:
        trycols = [q for q in now2 if q != a]
        mse2 = get_mse(trycols)
        if mse2 < smallest_mse:
            smallest_mse = mse2
            removeone = a
            now2.remove(removeone)
        print("backward variables", now2)


# Question 4
# This section loads the College data and prepares the response variable.
college = pd.read_csv("College.csv")
college = college.rename(columns={college.columns[0]: "CollegeName"})
college["Private"] = (college["Private"] == "Yes").astype(int)
five = ["Accept", "Enroll", "Outstate", "Books", "Grad.Rate"]
y = college["Room.Board"]

# Question 4(a)
pd.plotting.scatter_matrix(college[five + ["Room.Board"]], figsize=(10, 10))
plt.tight_layout()
plt.savefig("college_pairplot_studentish.png")
plt.close()
print("correlations")
print(college[five + ["Room.Board"]].corr()["Room.Board"])

# Question 4(b): best subset selection using the five proposed predictors.
print("best subset for five variables")
for num in range(1, 6):
    best = 999999999
    what = None
    for cols in combinations(five, num):
        reg = LinearRegression().fit(college[list(cols)], y)
        yh = reg.predict(college[list(cols)])
        mse1 = mean_squared_error(y, yh)
        if mse1 < best:
            best = mse1
            what = cols
    print(num, what, best)

# Question 4(d) and 4(e)
cols2 = [q for q in college.columns if q not in ["CollegeName", "Room.Board"]]
X = college[cols2]

# Split into training and testing data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.25, random_state=42
)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
lams = np.logspace(-4, 6, 100)
ridge_mse = []
lasso_mse = []
for lam in lams:
    rr = Ridge(alpha=lam)
    rr.fit(X_train, y_train)
    ridge_mse.append(mean_squared_error(y_test, rr.predict(X_test)))
    ll = Lasso(alpha=lam, max_iter=100000)
    ll.fit(X_train, y_train)
    lasso_mse.append(mean_squared_error(y_test, ll.predict(X_test)))

ridge_place = ridge_mse.index(min(ridge_mse))
lasso_place = lasso_mse.index(min(lasso_mse))

print("ridge lambda", lams[ridge_place])
print("ridge mspe", ridge_mse[ridge_place])
print("lasso lambda", lams[lasso_place])
print("lasso mspe", lasso_mse[lasso_place])
# Question 4(d): ridge MSPE plot.
plt.plot(lams, ridge_mse)
plt.xscale("log")
plt.xlabel("lambda")
plt.ylabel("MSPE")
plt.title("ridge")
plt.savefig("ridge_studentish.png")
plt.close()

# Question 4(e): Lasso MSPE plot.
plt.plot(lams, lasso_mse)
plt.xscale("log")
plt.xlabel("lambda")
plt.ylabel("MSPE")
plt.title("lasso")
plt.savefig("lasso_studentish.png")
plt.close()

# Question 4(f) is in the written answer.
