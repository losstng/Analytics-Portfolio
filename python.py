import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

####load
p = sns.load_dataset("penguins")
data = pd.read_csv('marketing_and_sales_data_evaluate_lr.csv')

####explore
p.head()
sns.pairplot(p)
d.isna().sum()
data[['TV','Radio','Social_Media']].describe()

missing_sales = data.Sales.isna().mean()
missing_sales = round(missing_sales*100, 2)
print('Percentage of promotions missing Sales: ' +  str(missing_sales) + '%')

print(data.groupby('TV')['Sales'].mean())

data = data.rename(columns={'Social Media': 'Social_Media'})
####cleaning
p = p[p['species'] != "Chinstrap"]

d = d.dropna(axis=0)
data = data.dropna(subset = ['Sales'], axis = 0)

p = p[["body_mass_g", "bill_length_mm", "sex", "species"]]
p.columns = ["body_mass_g", "bill_length_mm", "gender", "species"]
X_train, X_test, y_train, y_test = train_test_split(p_X, p_y, test_size=0.3, random_state=42)

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
