import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime, nltk, warnings
import matplotlib.cm as cm
import itertools
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import neighbors, linear_model, svm, tree, ensemble
from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from IPython.display import display, HTML
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)

#options to display all columns
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_columns', 3000)

# import data
# treat "CustomerID" and "InvoiceID" as string to preserve the leading zeros
df_initial = pd.read_csv('/Users/quentin/PycharmProjects/Kaggle-CustomerSegmentation/data/data.csv', encoding='ISO-8859-1',
                                        dtype={'CustomerID': str,'InvoiceID': str})

print('data shape:', df_initial.shape)
print('data info:', df_initial.info())

#convert InvoiceDate to datetime
df_initial['InvoiceDate'] = pd.to_datetime(df_initial['InvoiceDate'])

# provide infos on column types and number of null values
tab_info = pd.DataFrame(df_initial.dtypes).T.rename(index={0: 'column type'})
tab_info = pd.concat([tab_info, pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0: 'null values(nb)'})])
tab_info = pd.concat([tab_info, pd.DataFrame((df_initial.isnull().sum()/df_initial.shape[0])*100).T.rename(index={0: 'null values(%)'})])
print(tab_info)

print(df_initial.head())

df_initial.dropna(axis = 0, subset=['CustomerID'], inplace=True)
print('data shape:', df_initial.shape)
tab_info = pd.DataFrame(df_initial.dtypes).T.rename(index={0: 'column type'})
tab_info = pd.concat([tab_info, pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0: 'null values(nb)'})])
tab_info = pd.concat([tab_info, pd.DataFrame((df_initial.isnull().sum()/df_initial.shape[0])*100).T.rename(index={0: 'null values(%)'})])
print(tab_info)


