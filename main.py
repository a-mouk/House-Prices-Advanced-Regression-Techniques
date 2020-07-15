
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', 200)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

all_data.drop(['Id'], axis=1, inplace=True)

print(all_data.info())

count_empty = all_data.isnull().sum().sum()
print("Οι φαινομενικά ελλείπουσες τιμές είναι "+str(count_empty))

count_of_index = len(all_data.index)
count_of_col = len(all_data.columns)
count_of_total = count_of_index * count_of_col
print("Τα κελιά με περιεχόμενο είναι "+str(count_of_total))

per_empty = 100 * ( count_empty/count_of_total )
completeness = 100 - per_empty
compl_rounded = round(completeness, 2)

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
total = all_data.isnull().sum(axis = 0).sort_values(ascending=False)
missing_data = pd.concat([all_data_na, total, all_data.dtypes], axis=1, keys=['Missing Ratio', 'Total', 'Type'], sort=False)

print(missing_data.head(20))


f, ax = plt.subplots(figsize=(7, 5))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
plt.show()

all_data["PoolQC"] = all_data["PoolQC"].fillna("No")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("No")
all_data["Alley"] = all_data["Alley"].fillna("No")
all_data["Fence"] = all_data["Fence"].fillna("No")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("No")

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('No')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('No')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("No")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("No")

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()

print(all_data)

all_data_1 = all_data.copy()

all_data_notObj = all_data_1.select_dtypes(exclude=['object'])

all_data_Objs = all_data_1.select_dtypes(exclude=['float64', 'int64'])
print(all_data_Objs)

stat = all_data_1.describe()
print_full(stat)


print(all_data_1[['ExterQual']])
ExGd = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
        'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
for i in ExGd:
    all_data_1[i] = all_data_1[i].replace('Ex', 5).replace('Gd', 4).replace('TA', 3).replace('Fa', 2).replace('Po', 1).replace('No', 0)
print("Μετά τη μετατροπή:")
print(all_data_1[['ExterQual']])


all_data_Objs1 = all_data_1.select_dtypes(exclude=['float64', 'int64'])
print(all_data_Objs1)


all_data_1 = all_data_1.select_dtypes(exclude=['object'])


all_data_2 = all_data_1.copy()


all_data_1.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.show()

all_data_1.drop(['2ndFlrSF','3SsnPorch','BsmtFinSF2','BsmtHalfBath','EnclosedPorch',
                 'GarageYrBlt','KitchenAbvGr','LowQualFinSF','MiscVal','PoolArea',
                 'ScreenPorch','BsmtCond','ExterCond','GarageCond','GarageQual','PoolQC', 'MSSubClass'
], axis=1, inplace=True)


corrmat = all_data_1.corr()
plt.subplots(figsize=(20,17))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()

corr = all_data_1.corr()

fig, (ax) = plt.subplots(1, 1, figsize=(20,12))
hm = sns.heatmap(corr, 
                 ax=ax,           
                 cmap="coolwarm", # χρώμα χάρτη
                 annot=True, 
                 fmt='.2f',      
                 linewidths=.05)
fig.subplots_adjust(top=0.93)
plt.show()

corr = corr.abs()
corr['corr'] = corr.sum(axis=1)
rank_A = corr[['corr']].sort_values('corr', ascending=False)

rank_B = all_data_1.std(axis = 0).rename_axis('variable').to_frame('std')
rank_B = rank_B.sort_values('std', ascending=False)
print(rank_B)

objs = list(all_data_Objs1)

all_data_Objs2 = all_data_Objs1.copy()

from sklearn.preprocessing import LabelEncoder
for c in objs:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data_Objs1[c].values)) 
    all_data_Objs1[c] = lbl.transform(list(all_data_Objs1[c].values))

all_data_1.reset_index(drop=True, inplace=True)
all_data_Objs1.reset_index(drop=True, inplace=True)

all_data_all_are_num = pd.concat( [all_data_1, all_data_Objs1], axis=1)
print(all_data_all_are_num)
print(all_data_all_are_num.info())

all_data_Objs1.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.show()

all_data_all_are_num.drop([
                 'Alley','BldgType', 'BsmtFinType2','CentralAir','Condition1',
                 'Condition2','Electrical','Functional',
                 'Heating','PavedDrive','RoofMatl','Street'
], axis=1, inplace=True)

all_data_Objs2.drop(['Alley','BldgType', 'BsmtFinType2','CentralAir','Condition1',
                 'Condition2','Electrical','Functional',
                 'Heating','PavedDrive','RoofMatl','Street'
], axis=1, inplace=True)

objs2 = list(all_data_Objs2)
OneHotObjs = pd.get_dummies(all_data_Objs2, prefix=objs2)
OneHotObjs_2  = OneHotObjs.copy()

all_data_1.reset_index(drop=True, inplace=True)
OneHotObjs.reset_index(drop=True, inplace=True)

all_data_OH = pd.concat( [all_data_1, OneHotObjs], axis=1)
print(all_data_OH)

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

cols_list = list(all_data_2)

scaler = preprocessing.StandardScaler()
all_data_norm = scaler.fit_transform(all_data_2)
all_data_norm = pd.DataFrame(all_data_norm, columns = cols_list)

#all_data_norm = (all_data_2-all_data_2.mean())/all_data_2.std(ddof=0)
#all_data_norm

df_num = all_data_norm.select_dtypes(include = ['float64', 'int64'])
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.show()

all_data_norm.drop(['2ndFlrSF','3SsnPorch','BsmtFinSF2','BsmtHalfBath','EnclosedPorch',
                    'GarageYrBlt','KitchenAbvGr','LowQualFinSF','MiscVal','PoolArea',
                    'ScreenPorch','BsmtCond','ExterCond','GarageCond','GarageQual',
                    'PoolQC', 'MSSubClass'
], axis=1, inplace=True)
print("Τελικός πίνακας κανονικοποιημένων με z-score μεταβλητών:")
print(all_data_norm)


corr_norm = all_data_norm.corr()
corr_norm = corr_norm.abs()

corr_norm['corr'] = corr_norm.sum(axis=1)
rank_C = corr_norm[['corr']].sort_values('corr', ascending=False)
print(rank_C)

all_data_norm.drop([
'WoodDeckSF','HalfBath','LotArea','BsmtFullBath',
'BedroomAbvGr',
'OpenPorchSF',
'MoSold','YrSold','OverallCond'], axis=1)

rank_D = all_data_norm.std(axis = 0).rename_axis('variable').to_frame('std')
rank_D = rank_D.sort_values('std', ascending=False)
print(rank_D)

all_data_norm.reset_index(drop=True, inplace=True)
OneHotObjs_2.reset_index(drop=True, inplace=True)

all_data_OH_norm = pd.concat( [all_data_norm, OneHotObjs_2], axis=1)
print(all_data_OH_norm)



print("all_data_OH_norm")
print(all_data_OH_norm)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# silhouette
for n_clusters in range(2, 10):
    clusterer = KMeans (n_clusters=n_clusters)
    preds = clusterer.fit_predict(all_data_OH_norm)
    centers = clusterer.cluster_centers_

    score = silhouette_score (all_data_OH_norm, preds, metric='euclidean')
    print ("Για n_clusters = {}, το silhouette score είναι {})".format(n_clusters, score))

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(all_data_OH_norm)
    sse[k] = kmeans.inertia_ 

plt.figure(1, figsize = (15, 10))
plt.title(' Elbow  k-means')
plt.xlabel('k'); plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()

kmeans = KMeans(n_clusters=2)

y = kmeans.fit_predict(all_data_OH_norm)

all_data_OH_norm['Cluster'] = y
all_data_OH_norm[['Cluster']]

_id = []
for i in range(1,2920):
    _id.append(i)
all_data_OH_norm['id'] = pd.Series(_id)
all_data_OH_norm[['id','Cluster']]

cluster_0 = all_data_OH_norm[all_data_OH_norm["Cluster"] == 0]
cluster_1 = all_data_OH_norm[all_data_OH_norm["Cluster"] == 1]

cluster_0.drop(['Cluster'], axis=1, inplace=True)
cluster_1.drop(['Cluster'], axis=1, inplace=True)

print(cluster_0.head(3))

train_i = train[['Id','SalePrice']]

c_0 = pd.merge(cluster_0, train_i, left_on='id', right_on='Id', how='left')
c_0.drop(['Id'], axis=1, inplace=True)
print(c_0)

c_1 = pd.merge(cluster_1, train_i, left_on='id', right_on='Id', how='left')
c_1.drop(['Id'], axis=1, inplace=True)


train_c0 = c_0[c_0["id"] <= 1460]
test_c0 = c_0[c_0["id"] > 1460]
train_c1 = c_1[c_1["id"] <= 1460]
test_c1 = c_1[c_1["id"] > 1460]

#train_c0_id = train_c0[['id']]
test_c0_id = test_c0[['id']]
train_c0.drop(['id'], axis=1, inplace=True)
test_c0.drop(['id'], axis=1, inplace=True)
test_c0.drop(['SalePrice'], axis=1, inplace=True)

#train_c1_id = train_c1[['id']]
test_c1_id = test_c1[['id']]
train_c1.drop(['id'], axis=1, inplace=True)
test_c1.drop(['id'], axis=1, inplace=True)
test_c1.drop(['SalePrice'], axis=1, inplace=True)

test_c0.head(3)

from sklearn.model_selection import train_test_split
from sklearn import ensemble
import sklearn.metrics as metrics

#  Gradient Boosting Regression
def GBR(train_c, test_c, test_c_id):
 
    X_train, X_test, y_train, y_test = train_test_split(train_c.drop('SalePrice', axis=1), 
                                                        train_c['SalePrice'], test_size=0.3, random_state=101)
    y_train= y_train.values.reshape(-1,1)
    y_test= y_test.values.reshape(-1,1)
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)
    y_train = sc_X.fit_transform(y_train)
    y_test = sc_y.fit_transform(y_test)
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(X_train, y_train)
    clf_pred=clf.predict(X_test)
    clf_pred= clf_pred.reshape(-1,1)

    print("Mετρικές σφαλμάτων για τη  συστάδα:")
    print('MAE:', metrics.mean_absolute_error(y_test, clf_pred))
    print('MSE:', metrics.mean_squared_error(y_test, clf_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, clf_pred)))
    plt.figure(figsize=(15,8))
    plt.scatter(y_test,clf_pred, c= 'brown')
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')
    plt.show()

    test_c = sc_X.fit_transform(test_c)
    test_prediction_clf= clf.predict(test_c)
    test_prediction_clf= test_prediction_clf.reshape(-1,1)
    test_prediction_clf =sc_y.inverse_transform(test_prediction_clf)

    test_prediction_clf = pd.DataFrame(test_prediction_clf, columns=['SalePrice'])
    test_c_id.reset_index(drop=True, inplace=True)
    test_prediction_clf.reset_index(drop=True, inplace=True)
    result = pd.concat( [test_c_id, test_prediction_clf], axis=1)
    
    return result

result_0 = GBR(train_c0, test_c0, test_c0_id)
result_1 = GBR(train_c1, test_c1, test_c1_id)

frames = [result_0, result_1]
sub = pd.concat(frames)

sub1 = sub.sort_values("id", ascending = True)

sub1.columns = ['Id', 'SalePrice']

sub1.to_csv('submission.csv',index=False)
