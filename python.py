import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
%pylab inline
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn import naive_bayes
from sklearn import model_selection
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import PredefinedSplit

#####load & data load
file = 'Churn_Modelling.csv'
df_original = pd.read_csv(file)
df_original.head()

p = sns.load_dataset("penguins")
data = pd.read_csv('marketing_and_sales_data_evaluate_lr.csv')
img = plt.imread('using_kmeans_for_color_compression_tulips_photo.jpg')
rng = np.random.default_rng(seed=44) # random

#####explore
p.head()
df.dtypes
sns.pairplot(p)
d.isna().sum()
data[['TV','Radio','Social_Media']].describe()
colorless["color"].values
df_original['Class'].unique()
df['your_column'].value_counts()
df['your_column'].value_counts(normalize=True) * 100

df_original.isna().sum()

missing_sales = data.Sales.isna().mean()
missing_sales = round(missing_sales*100, 2)
print('Percentage of promotions missing Sales: ' +  str(missing_sales) + '%')



print(data.groupby('TV')['Sales'].mean())

sns.boxplot(x = "color", y = "log_price", data = d)

avg_churned_bal = df_original[df_original['Exited']==1]['Balance'].mean()
avg_churned_bal

#####cleaning
p = p[p['species'] != "Chinstrap"]
df.drop('column_name', axis=1, inplace=True)

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

colorless.to_csv('diamonds.csv',index=False,header=list(colorless.columns)

## scaling
X_scaled = StandardScaler().fit_transform(X)
X_scaled[:2,:]

## Prepping data, feature engineering

penguins_subset['sex'] = penguins_subset['sex'].str.upper()

# Convert `sex` column from categorical to numeric.
penguins_subset = pd.get_dummies(penguins_subset, drop_first = True, columns=['sex'])

#### feature engineering

# selection
churn_df = df_original.drop(['RowNumber', 'CustomerId', 'Surname', 'Gender'], 
                            axis=1)

# transformation

df_subset['Class'] = df_subset['Class'].map({"Business": 3, "Eco Plus": 2, "Eco": 1}) # manually assign values
churn_df = pd.get_dummies(churn_df, drop_first=True)

# splitting the data
y = churn_df['Exited']

X = churn_df.copy()
X = X.drop('Exited', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25, stratify=y, 
                                                    random_state=42)

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

##### randomly generating numbers

rng = np.random.default_rng(seed=44)
centers = rng.integers(low=3, high=7)
X, y = make_blobs(n_samples=1000, n_features=6, centers=centers, random_state=42)
X = pd.DataFrame(X)
X.head()



#####ordinary least squared
from statsmodels.formula.api import ols

ols_d = p_f[['bill_length_mm', 'body_mass_g']]
ols_formula = "body_mass_g ~ bill_length_mm"

OLS = ols(formula=ols_formula, data=ols_d)
model=OLS.fit()
model.summary()

#####models assumptions
#linearity assumption
sns.regplot(x = "bill_length_mm", y = "body_mass_g", data = ols_d) 

##normality assumption
fitted_values = model.predict(X) 
r = model.resid
fig = sns.histplot(r)
fig.set_xlabel('Residual Value')
fig.set_title('Histogram of Residuals')
plt.show()
&&&&
fig = sm.qqplot(model.resid, line = 's')
plt.show()

##homoscedasticity assumption
fig = sns.scatterplot(x=fitted_values, y=residuals)
fig.axhline(0)

fig.set_xlabel("Fitted Values")
fig.set_ylabel("Residuals")
plt.show()

##no multicollinearity (applicable to multiple linear)
from statsmodels.stats.outliers_influence import variance_inflation_factor
VIF
X = data[['Radio','Social_Media']]
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
df_vif = pd.DataFrame(vif, index=X.columns, columns = ['VIF'])
df_vif

#####multiple linear regression
p_X = p[["bill_length_mm", "gender", "species"]]
p_y = p[["body_mass_g"]]

ols_formula = 'body_mass_g ~ bill_length_mm + C(gender) + C(species)'

OLS = ols(formula = ols_formula, data = p_X)
model=OLS.fit()
model.summary()

#####ANOVA & ANCOVA & MANOVA & MANCOVA

###One way Anova
model = ols(formula = "log_price ~ C(color)", data = d.fit())

sm.stats.anova_lm(model, typ = 2)
sm.stats.anova_lm(model, typ = 1)
sm.stats.anova_lm(model, typ = 3)

###Two way Anova
model2 = ols(formula = "log_price ~ C(color) + C(cut) + C(color):C(cut)", data = diamonds2).fit()

sm.stats.anova_lm(model2, typ = 2)
sm.stats.anova_lm(model2, typ = 1)
sm.stats.anova_lm(model2, typ = 3)

###ANOVA post hoc test - used for hypothesis testing
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey_oneway = pairwise_tukeyhsd(endog = diamonds["log_price"], groups = diamonds["color"], alpha = 0.05)
tukey_oneway.summary()

###ANCOVA
model_ancova = ols("log_price ~ C(color) + carat", data=df_ancova).fit()
sm.stats.anova_lm(model_ancova, typ=2)

####MANOVA
from statsmodels.multivariate.manova import MANOVA

manova_model = MANOVA.from_formula('Sales + Revenue ~ C(Channel)', data=df_manova)
print(manova_model.mv_test())

###MANCOVA
mancova_model = MANOVA.from_formula('Sales + Revenue ~ C(Channel) + MarketingSpend', data=df_mancova)
print(mancova_model.mv_test())

###### Logistic regression
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

### Confusion metrics
cm = metrics.confusion_matrix(y_test, y_pred, labels = clf.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = clf.classes_)
disp.plot()

##### K-means
## image reference
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

## color conversion helper
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

#### standard reference
kmeans3 = KMeans(n_clusters=3, random_state=42)

kmeans3.fit(X_scaled)

## results & metrics

# inertia
print('Clusters: ', kmeans3.labels_)
print('Inertia: ', kmeans3.inertia_)

# Create a list from 2-10. 
num_clusters = [i for i in range(2, 11)]

def kmeans_inertia(num_clusters, x_vals):
    inertia = []
    for num in num_clusters:
        kms = KMeans(n_clusters=num, random_state=42)
        kms.fit(x_vals)
        inertia.append(kms.inertia_)
    
    return inertia
  
inertia = kmeans_inertia(num_clusters, X_scaled)
inertia

# silhoutte score
kmeans3_sil_score = silhouette_score(X_scaled, kmeans3.labels_)
kmeans3_sil_score

def kmeans_sil(num_clusters, x_vals):
    '''
    Fits a KMeans model for different values of k.
    Calculates a silhouette score for each k value

    Args:
        num_clusters: (list of ints)  - The different k values to try
        x_vals:       (array)         - The training data

    Returns: 
        sil_score:    (list)          - A list of silhouette scores, one for each \
                                      value of k
    '''
  
    sil_score = []
    for num in num_clusters:
        kms = KMeans(n_clusters=num, random_state=42)
        kms.fit(x_vals)
        sil_score.append(silhouette_score(x_vals, kms.labels_))
    
    return sil_score

sil_score = kmeans_sil(num_clusters, X_scaled)
sil_score

## visualization with K-means metrics
plot = sns.lineplot(x=num_clusters, y=inertia, marker = 'o')
plot.set_xlabel("Number of clusters");
plot.set_ylabel("Inertia");

plot = sns.lineplot(x=num_clusters, y=sil_score)
plot.set_xlabel("# of clusters");
plot.set_ylabel("Silhouette Score");

centers

## further analysis

kmeans5 = KMeans(n_clusters=5, random_state=42)
kmeans5.fit(X_scaled)

print(kmeans5.labels_[:5])
print('Unique labels:', np.unique(kmeans5.labels_))

X['cluster'] = kmeans5.labels_
X.head()

# Verify if any `cluster` can be differentiated by `species`.
penguins_subset.groupby(by=['cluster', 'species']).size()

# visualization
penguins_subset.groupby(by=['cluster', 'species']).size().plot.bar(title='Clusters differentiated by species',
                                                                   figsize=(6, 5),
                                                                   ylabel='Size',
                                                                   xlabel='(Cluster, Species)');

penguins_subset.groupby(by=['cluster','species','sex_MALE']).size().unstack(level = 'species', fill_value=0).plot.bar(title='Clusters differentiated by species and sex',
                                                                                                                      figsize=(6, 5),
                                                                                                                      ylabel='Size',
                                                                                                                      xlabel='(Cluster, Sex)')
plt.legend(bbox_to_anchor=(1.3, 1.0))

##### Naive Bayes Model
nb = naive_bayes.GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print('accuracy score:'), print(metrics.accuracy_score(y_test, y_pred))
print('precision score:'), print(metrics.precision_score(y_test, y_pred))
print('recall score:'), print(metrics.recall_score(y_test, y_pred))
print('f1 score:'), print(metrics.f1_score(y_test, y_pred))


##### Decision Tree
#### baseline model
decision_tree = DecisionTreeClassifier(random_state=0)

decision_tree.fit(X_train, y_train)

dt_pred = decision_tree.predict(X_test)

## small cap performance
print("Accuracy:", "%.3f" % accuracy_score(y_test, dt_pred))
print("Precision:", "%.3f" % precision_score(y_test, dt_pred))
print("Recall:", "%.3f" % recall_score(y_test, dt_pred))
print("F1 Score:", "%.3f" % f1_score(y_test, dt_pred))

## confusion matrix
def conf_matrix_plot(model, x_data, y_data):
  
    model_pred = model.predict(x_data)
    cm = confusion_matrix(y_data, model_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=model.classes_)
  
    disp.plot(values_format='')  # `values_format=''` suppresses scientific notation
    plt.show()

conf_matrix_plot(decision_tree, X_test, y_test)

## decision tree
plt.figure(figsize=(15,12))
plot_tree(decision_tree, max_depth=2, fontsize=14, feature_names=X.columns, 
          class_names={0:'stayed', 1:'churned'}, filled=True);
plt.show()


plt.figure(figsize=(20,12))
plot_tree(clf.best_estimator_, max_depth=2, fontsize=14, feature_names=X.columns);

## hyperparameter tuning
# Assign a dictionary of hyperparameters to search over
tree_para = {'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50],
             'min_samples_leaf': [2, 5, 10, 20, 50]}

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1'}

tuned_decision_tree = DecisionTreeClassifier(random_state = 42)

importances = decision_tree.feature_importances_

forest_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax);

# Instantiate the GridSearch
clf = GridSearchCV(tuned_decision_tree, 
                   tree_para, 
                   scoring = scoring, 
                   cv=5, 
                   refit="f1")

# Fit the model
clf.fit(X_train, y_train)

# Examine the best model from GridSearch
clf.best_estimator_

print("Best Avg. Validation Score: ", "%.4f" % clf.best_score_)

## results
def make_results(model_name, model_object):
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
  
    # Create table of results
    table = pd.DataFrame({'Model': [model_name],
                        'F1': [f1],
                        'Recall': [recall],
                        'Precision': [precision],
                        'Accuracy': [accuracy]
                         }
                        )
  
    return table
result_table = make_results("Tuned Decision Tree", clf)
result_table.to_csv("Results.csv")
result_table

# adding further results

results = pd.concat([rf_cv_results, results])
results

## further tuning in Bagging & Random Forest
rf = RandomForestClassifier(random_state=0)

cv_params = {'max_depth': [2,3,4,5, None], 
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'max_features': [2,3,4],
             'n_estimators': [75, 100, 125, 150]
             }  

scoring = {'accuracy', 'precision', 'recall', 'f1'}

rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=5, refit='f1')
rf_cv.fit(X_train, y_train)

rf_cv.best_params_

rf_cv.best_score_

## further process in seperate validation set

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, 
                                            stratify=y_train, random_state=10)
split_index = [0 if x in X_val.index else -1 for x in X_train.index]

rf = RandomForestClassifier(random_state=0)

cv_params = {'max_depth': [2,3,4,5, None], 
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'max_features': [2,3,4],
             'n_estimators': [75, 100, 125, 150]
             }  

scoring = {'accuracy', 'precision', 'recall', 'f1'}

custom_split = PredefinedSplit(split_index)

rf_val = GridSearchCV(rf, cv_params, scoring=scoring, cv=custom_split, refit='f1')

rf_val.fit(X_train, y_train)

rf_val.best_params_

##### ML model saving
path = '/home/jovyan/work/'
with open(path+'rf_cv_model.pickle', 'wb') as to_write:
    pickle.dump(rf_cv, to_write)

with open(path + 'rf_cv_model.pickle', 'rb') as to_read:
    rf_cv = pickle.load(to_read)

