import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

####load
p = sns.load_dataset("penguins")
data = pd.read_csv('marketing_and_sales_data_evaluate_lr.csv')

####explore
p.head()
sns.pairplot(p)
d.isna().sum()
data[['TV','Radio','Social_Media']].describe()
colorless["color"].values

missing_sales = data.Sales.isna().mean()
missing_sales = round(missing_sales*100, 2)
print('Percentage of promotions missing Sales: ' +  str(missing_sales) + '%')

print(data.groupby('TV')['Sales'].mean())

sns.boxplot(x = "color", y = "log_price", data = d)

####cleaning
p = p[p['species'] != "Chinstrap"]

colorless = d[d["color"].isin(["E","F","H","D","I"])]
colorless = colorless[["color","price"]].reset_index(drop=True)
colorless.color = colorless.color.cat.remove_categories(["G","J"])
colorless.insert(2, "log_price", [math.log(price) for price in colorless["price"]])

d = d.dropna(axis=0)
data = data.dropna(subset = ['Sales'], axis = 0)

p = p[["body_mass_g", "bill_length_mm", "sex", "species"]]
p.columns = ["body_mass_g", "bill_length_mm", "gender", "species"]
X_train, X_test, y_train, y_test = train_test_split(p_X, p_y, test_size=0.3, random_state=42)

data = data.rename(columns={'Social Media': 'Social_Media'})

colorless.to_csv('diamonds.csv',index=False,header=list(colorless.columns))

####ordinary least squared
from statsmodels.formula.api import ols

ols_d = p_f[['bill_length_mm', 'body_mass_g']]
ols_formula = "body_mass_g ~ bill_length_mm"

OLS = ols(formula=ols_formula, data=ols_d)
model=OLS.fit()
model.summary()

###models assumptions
#linearity assumption
sns.regplot(x = "bill_length_mm", y = "body_mass_g", data = ols_d) 

#normality assumption
fitted_values = model.predict(X) 
r = model.resid
fig = sns.histplot(r)
fig.set_xlabel('Residual Value')
fig.set_title('Histogram of Residuals')
plt.show()
&&&&
fig = sm.qqplot(model.resid, line = 's')
plt.show()

#homoscedasticity assumption
fig = sns.scatterplot(x=fitted_values, y=residuals)
fig.axhline(0)

fig.set_xlabel("Fitted Values")
fig.set_ylabel("Residuals")
plt.show()

#no multicollinearity (applicable to multiple linear)
from statsmodels.stats.outliers_influence import variance_inflation_factor
VIF
X = data[['Radio','Social_Media']]
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
df_vif = pd.DataFrame(vif, index=X.columns, columns = ['VIF'])
df_vif

####multiple linear regression
p_X = p[["bill_length_mm", "gender", "species"]]
p_y = p[["body_mass_g"]]

ols_formula = 'body_mass_g ~ bill_length_mm + C(gender) + C(species)'

OLS = ols(formula = ols_formula, data = p_X)
model=OLS.fit()
model.summary()

####ANOVA & ANCOVA & MANOVA & MANCOVA

#One way Anova
model = ols(formula = "log_price ~ C(color)", data = d.fit())

sm.stats.anova_lm(model, typ = 2)
sm.stats.anova_lm(model, typ = 1)
sm.stats.anova_lm(model, typ = 3)

#Two way Anova
model2 = ols(formula = "log_price ~ C(color) + C(cut) + C(color):C(cut)", data = diamonds2).fit()

sm.stats.anova_lm(model2, typ = 2)
sm.stats.anova_lm(model2, typ = 1)
sm.stats.anova_lm(model2, typ = 3)

#ANOVA post hoc test - used for hypothesis testing
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey_oneway = pairwise_tukeyhsd(endog = diamonds["log_price"], groups = diamonds["color"], alpha = 0.05)
tukey_oneway.summary()

#ANCOVA
model_ancova = ols("log_price ~ C(color) + carat", data=df_ancova).fit()
sm.stats.anova_lm(model_ancova, typ=2)

#MANOVA
from statsmodels.multivariate.manova import MANOVA

manova_model = MANOVA.from_formula('Sales + Revenue ~ C(Channel)', data=df_manova)
print(manova_model.mv_test())

#MANCOVA
mancova_model = MANOVA.from_formula('Sales + Revenue ~ C(Channel) + MarketingSpend', data=df_mancova)
print(mancova_model.mv_test())

#### Logistic regression
# binomial logistic regression model
X = a[['Acc (vertical)']]
Y = a[['LyingDown']]

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

clf = LogisticRegression().fit(X_train,y_train)

clf.coef_
clf.intercept_
y_pred = clf.predict(X_test)
clf.predict_proba(X_test)[::,-1]

sns.regplot(x='Acc (vertical)', y='LyingDown', data=a, logistic=True)

# Confusion metrics
cm = metrics.confusion_matrix(y_test, y_pred, labels = clf.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = clf.classes_)
disp.plot()
