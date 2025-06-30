import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
%pylab inline
import plotly.graph_objects as go
from sklearn.cluster import KMeans

####load
p = sns.load_dataset("penguins")
data = pd.read_csv('marketing_and_sales_data_evaluate_lr.csv')
img = plt.imread('using_kmeans_for_color_compression_tulips_photo.jpg')

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

#####photos
img = plt.imread('using_kmeans_for_color_compression_tulips_photo.jpg')

## Reshape the image so that each row represents a single pixel | defined by three values: R, G, B
print(img.shape)
plt.imshow(img)
plt.axis('off')

img_flat = img.reshape(img.shape[0]*img.shape[1], 3)
img_flat[:5, :]

img_flat_df = pd.DataFrame(img_flat, columns = ['r', 'g', 'b'])
img_flat_df.head()

# Create 3D plot where each pixel in the `img` is displayed in its actual color
trace = go.Scatter3d(x = img_flat_df.r,
                     y = img_flat_df.g,
                     z = img_flat_df.b,
                     mode='markers',
                     marker=dict(size=1,
                                 color=['rgb({},{},{})'.format(r,g,b) for r,g,b 
                                        in zip(img_flat_df.r.values, 
                                               img_flat_df.g.values, 
                                               img_flat_df.b.values)],
                                 opacity=0.5))

data = [trace]

layout = go.Layout(margin=dict(l=0,
                               r=0,
                               b=0,
                               t=0),
                               )

fig = go.Figure(data=data, layout=layout)
fig.update_layout(scene = dict(
                    xaxis_title='R',
                    yaxis_title='G',
                    zaxis_title='B'),
                  )
fig.show()

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

####models assumptions
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

#### K-means
kmeans = KMeans(n_clusters=3, random_state=42).fit(img_flat)
np.unique(kmeans3.labels_)

centers = kmeans3.cluster_centers_

def show_swatch(RGB_value):
    '''
    Takes in an RGB value and outputs a color swatch
    '''
    R, G, B = RGB_value
    rgb = [[np.array([R,G,B]).astype('uint8')]]
    plt.figure()
    plt.imshow(rgb)
    plt.axis('off');

for pixel in centers:
    show_swatch(pixel)

def cluster_image(k, img=img):
    '''
    Fits a K-means model to a photograph.
    Replaces photo's pixels with RGB values of model's centroids.
    Displays the updated image.

    Args:
      k:    (int)          - Your selected K-value
      img:  (numpy array)  - Your original image converted to a numpy array

    Returns:
      The output of plt.imshow(new_img), where new_img is a new numpy array \
      where each row of the original array has been replaced with the \ 
      coordinates of its nearest centroid.
    '''

    img_flat = img.reshape(img.shape[0]*img.shape[1], 3)
    kmeans = KMeans(n_clusters = k, random_state = 42).fit(img_flat)
    new_img = img_flat.copy()
  
    for i in np.unique(kmeans.labels_):
        new_img[kmeans.labels_ == i, :] = kmeans.cluster_centers_[i]
  
    new_img = new_img.reshape(img.shape)

    return plt.imshow(new_img), plt.axis('off');

cluster_image(3);


print(kmeans3.labels_.shape)
print(kmeans3.labels_)
print(np.unique(kmeans3.labels_))
print(kmeans3.cluster_centers_)

img_flat_df['cluster'] = kmeans3.labels_
img_flat_df.head()

# color conversion helper
series_conversion = {0: 'rgb' +str(tuple(kmeans3.cluster_centers_[0])),
                     1: 'rgb' +str(tuple(kmeans3.cluster_centers_[1])),
                     2: 'rgb' +str(tuple(kmeans3.cluster_centers_[2])),
                     }
series_conversion

# Replace the cluster numbers in the 'cluster' col with formatted RGB values 
# (made ready for plotting)
img_flat_df['cluster'] = img_flat_df['cluster'].map(series_conversion)
img_flat_df.head()

# show the data plot when k=3
trace = go.Scatter3d(x = img_flat_df.r,
                     y = img_flat_df.g,
                     z = img_flat_df.b,
                     mode='markers',
                     marker=dict(size=1,
                                 color=img_flat_df.cluster,
                                 opacity=1))

data = trace

layout = go.Layout(margin=dict(l=0,
                               r=0,
                               b=0,
                               t=0))

fig = go.Figure(data=data, layout=layout)
fig.show()


# now the function that will guides us all
def cluster_image_grid(k, ax, img=img):
    '''
    Fits a K-means model to a photograph.
    Replaces photo's pixels with RGB values of model's centroids.
    Displays the updated image on an axis of a figure.

    Args:
      k:    (int)          - Your selected K-value
      ax:   (int)          - Index of the axis of the figure to plot to
      img:  (numpy array)  - Your original image converted to a numpy array

    Returns:
      A new image where each row of img's array has been replaced with the \ 
      coordinates of its nearest centroid. Image is assigned to an axis that \
      can be used in an image grid figure.
    '''
    img_flat = img.reshape(img.shape[0]*img.shape[1], 3)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(img_flat)
    new_img = img_flat.copy()

    for i in np.unique(kmeans.labels_):
        new_img[kmeans.labels_==i, :] = kmeans.cluster_centers_[i]

    new_img = new_img.reshape(img.shape)
    ax.imshow(new_img)
    ax.axis('off')

fig, axs = plt.subplots(3, 3)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(9, 12)
axs = axs.flatten()
k_values = np.arange(2, 11)
for i, k in enumerate(k_values):
    cluster_image_grid(k, axs[i], img=img)
    axs[i].title.set_text('k=' + str(k))
