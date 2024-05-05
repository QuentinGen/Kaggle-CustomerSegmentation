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
import plotly.graph_objs as go
from plotly.offline import plot

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

# remove nulls and re-display info
df_initial.dropna(axis = 0, subset=['CustomerID'], inplace=True)
print('data shape:', df_initial.shape)
tab_info = pd.DataFrame(df_initial.dtypes).T.rename(index={0: 'column type'})
tab_info = pd.concat([tab_info, pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0: 'null values(nb)'})])
tab_info = pd.concat([tab_info, pd.DataFrame((df_initial.isnull().sum()/df_initial.shape[0])*100).T.rename(index={0: 'null values(%)'})])
print(tab_info)

# check for duplicates
print('duplicate values:', df_initial.duplicated().sum())
df_initial.drop_duplicates(inplace=True)

# count the number of countries
# Group the DataFrame by 'CustomerID', 'InvoiceNo', and 'Country' columns,
# and count the occurrences of each combination
temp = df_initial[['CustomerID','InvoiceNo','Country']].groupby(['CustomerID', 'InvoiceNo', 'Country']).count()
temp = temp.reset_index(drop=False)
countries = temp['Country'].value_counts()
print('nb of countries in the dataframe:', len(countries))

# choropleth map
data = dict(
    type='choropleth',  # Type of plot: choropleth map
    locations=countries.index,  # Locations (countries) for the map
    locationmode='country names',  # Location mode: specify that the locations are country names
    z=countries,  # Values to be visualized (order counts for each country)
    text=countries.index,  # Text to be displayed when hovering over each country
    colorbar={'title': 'Order nb.'},  # Colorbar title
    colorscale=[  # Color scale for the map
        [0, 'rgb(224,255,255)'],
        [0.01, 'rgb(166,206,227)'],
        [0.02, 'rgb(31,120,180)'],
        [0.03, 'rgb(178,223,138)'],
        [0.05, 'rgb(51,160,44)'],
        [0.10, 'rgb(251,154,153)'],
        [0.20, 'rgb(255,255,0)'],
        [1, 'rgb(227,26,28)']],
reversescale=False)

layout = dict(title='Number of order per country',
geo = dict(showframe=True, projection={'type': 'mercator'}))

choromap = go.Figure(data = [data], layout=layout)
iplot(choromap, validate=False)
plot(choromap, filename='choropleth_map.html', auto_open=False)

# unique number of users and products
counts = pd.DataFrame([{'products': df_initial['StockCode'].nunique(),
               'transactions': df_initial['InvoiceNo'].nunique(),
               'customers': df_initial['CustomerID'].nunique(),
               }],columns=['products', 'transactions', 'customers'], index= ['quantity'])

print(counts)

# number of product purchased in every transaction

# Group the DataFrame by 'CustomerID' and 'InvoiceNo' columns,
# and count the occurrences of 'InvoiceDate' for each group
temp = df_initial.groupby(['CustomerID','InvoiceNo'], as_index=False)['InvoiceDate'].count()

# Rename the column 'InvoiceDate' to 'Number of Products'
nb_products_per_basket = temp.rename(columns = {'InvoiceDate': 'Number of Products'})

# Display the first 10 rows of the DataFrame sorted by 'CustomerID'
print(nb_products_per_basket[:10].sort_values('CustomerID'))