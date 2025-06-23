import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

d = pd.read_csv('marketing_sales_data.csv')
d = sns.load_dataset("diamonds", cache=False)


d.head()
d["color"].value_counts()

# some data cleaning
colorless = d[d["color"].isin(["E","F","H","D","I"])]
colorless = colorless[["color","price"]].reset_index(drop=True)
colorless.color = colorless.color.cat.remove_categories(["G","J"])
colorless["color"].values
colorless.insert(2, "log_price", [math.log(price) for price in colorless["price"]])
colorless.dropna(inplace=True)
colorless.reset_index(inplace=True, drop=True)

colorless.to_csv('diamonds.csv',index=False,header=list(colorless.columns))

sns.boxplot(x = "color", y = "log_price", data = d)
sns.boxplot(x='TV',y='Sales',data=d)

d.isna().sum()
d= d.dropna(axis=0)
d.isna().sum()

sns.pairplot(d)


ols_f = 'Sales ~ C(TV)'

OLS = ols(formula = ols_f, data = d)

model = OLS.fit()

model_result = model.summary()

model_result

sns.pairplot(d)
r = model.resid

sns.pairplot(d)
r = model.resid

fig, axes = plt.subplots(1,2, figsize=(8,4))

sns.histplot(r, ax=axes[0])

axes[0].set_xlabel('Residual Value')

axes[0].set_title("Histogram of Residuals")

sm.qqplot(r, line='s', ax = axes[1])

axes[1].set_title('Normal QQ Plot')

plt.tight_layout()

plt.show()


fig = sns.scatterplot(x=model.fittedvalues, y=model.resid)

fig.set_xlabel('Fitted Values')

fig.set_ylabel('Residuals')

fig.set_title('Fitted Values v. Residuals')

fig.axhline(0)

plt.show()

sns.boxplot(x = "TV", y = "Sales", data = d)

print(f'''{sm.stats.anova_lm(model, typ=2)}

{sm.stats.anova_lm(model, typ=1)}

{sm.stats.anova_lm(model, typ=3)}''')

tukey_oneway = pairwise_tukeyhsd(endog=d['Sales'], groups= d['TV'])

tukey_oneway.summary()
