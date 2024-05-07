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

# Number of transactions corresponding to canceled orders:
condition = nb_products_per_basket['InvoiceNo'].str.startswith('C')
nb_products_per_basket['order_canceled'] = condition.astype(int)
print(nb_products_per_basket[:10].sort_values('CustomerID'))

# Calculate the percentage of canceled orders
total_orders = nb_products_per_basket.shape[0]
canceled_order = nb_products_per_basket['order_canceled'].sum()
percentage_canceled = canceled_order / total_orders * 100
print(f'Percentage of canceled orders:, {percentage_canceled:.2f}%')

# Check if there is always a corresponding positive order with the same quantity as negative order
# Filter the DataFrame 'df_initial' to include only the rows where the 'Quantity' column is less than 0,
# and select specific columns ('CustomerID', 'Quantity', 'StockCode', 'Description', 'UnitPrice').
df_check = df_initial[(df_initial['Quantity'] < 0) & (df_initial['Description'] != 'Discount')][['CustomerID', 'Quantity', 'StockCode', 'Description', 'UnitPrice']]

# Iterate over each row in the filtered DataFrame 'df_check'.
for index, col in df_check.iterrows():
    # Check if there are no rows in 'df_initial' that match the conditions specified.
    if df_initial[(df_initial['CustomerID'] == col[0]) & (df_initial['Quantity'] == -col[1])
            & (df_initial['Description'] == col[2])].shape[0] == 0:
        # If the conditions are not fulfilled, print the corresponding row from 'df_check'.
        print(df_check.loc[index])
        # Print a separator and a message indicating the hypothesis is not fulfilled.
        print(15*'-'+'>'+'HYPOTHESIS NOT FULFILLED')
        # Exit the loop.
        break

# Make a deep copy of the initial DataFrame to work with
df_cleaned = df_initial.copy(deep=True)

# Add a new column 'QuantityCanceled' initialized with zeros
df_cleaned['QuantityCanceled'] = 0

# Initialize empty lists to store indices of doubtful entries and entries to remove
entry_to_remove = []
doubtful_entry = []

# Loop over each row in the initial DataFrame
for index, col in df_initial.iterrows():
    # Check conditions for cancelations
    if (col['Quantity'] > 0) or (col['Description'] == 'Discount'):
        # Skip rows representing valid transactions (Quantity > 0) or discounts
        continue

    # Extract subset DataFrame of potential counterpart transactions
    df_test = df_initial[(df_initial['CustomerID'] == col['CustomerID']) &
                         (df_initial['StockCode'] == col['StockCode']) &
                         (df_initial['InvoiceDate'] < col['InvoiceDate']) &
                         (df_initial['Quantity'] > 0)].copy()

    # Cancelation WITHOUT counterpart
    if df_test.shape[0] == 0:
        # Add index of doubtful entry to list
        doubtful_entry.append(index)

    # Cancelation WITH a counterpart
    elif df_test.shape[0] == 1:
        # Get index of the counterpart transaction
        index_order = df_test.index[0]
        # Set QuantityCanceled to negative value of Quantity of the original transaction
        df_cleaned.loc[index_order, 'QuantityCanceled'] = -col['Quantity']
        # Mark the original transaction for removal
        entry_to_remove.append(index)

    # Various counterparts exist in orders: we delete the last one
    elif df_test.shape[0] > 1:
        # Sort the subset DataFrame by index in descending order (most recent first)
        df_test.sort_index(axis=0, ascending=False, inplace=True)
        # Iterate over potential counterpart transactions
        for ind, val in df_test.iterrows():
            # Check if the canceled quantity exceeds the available quantity
            if val['Quantity'] < -col['Quantity']:
                # If so, move to the next potential counterpart transaction
                continue
            # Update df_cleaned to reflect the canceled quantity
            df_cleaned.loc[ind, 'QuantityCanceled'] = -col['Quantity']
            # Mark the original transaction for removal
            entry_to_remove.append(index)
            # Exit the loop after processing the first valid counterpart transaction
            break

print('nb of entry to remove:', len(entry_to_remove))
print('nb of doubtfull entry:', len(doubtful_entry))

# Drop rows marked for removal from df_cleaned
df_cleaned.drop(entry_to_remove, axis=0, inplace=True)
# Drop doubtful entries from df_cleaned
df_cleaned.drop(doubtful_entry, axis=0, inplace=True)
# Select remaining entries with negative quantity and excluding discounts
remaining_entries = df_cleaned[(df_cleaned['Quantity'] < 0) & (df_cleaned['StockCode'] != 'D')]
# Print the number of entries to be deleted
print("Number of entries to delete: {}".format(remaining_entries.shape[0]))

# Print the first 5 entries to be deleted
print(remaining_entries.head(5))
# Example of filtering by specific conditions
print(df_cleaned[(df_cleaned['CustomerID'] == 14048) & (df_cleaned['StockCode'] == '22464')])

# Extract unique StockCodes containing only alphabetic characters
list_special_codes = df_cleaned[df_cleaned['StockCode'].str.contains('^[a-zA-Z]+', regex=True)]['StockCode'].unique()
print(list_special_codes)

# Return the description for each of those special codes
for code in list_special_codes:
    print(code, df_cleaned[df_cleaned['StockCode']==code]['Description'].unique())

# TotalPrice to indicates the total price of every purchase
df_cleaned['TotalPrice'] = (df_cleaned['Quantity'] - df_cleaned['QuantityCanceled']) * df_cleaned['UnitPrice']
df_cleaned.sort_values('CustomerID')[:5]

# Group all product purchases from an order and calculate total price
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
basket_price = temp.rename(columns = {'TotalPrice': 'Basket Price'})

# Order Date
df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
df_cleaned.drop('InvoiceDate_int',axis = 1, inplace=True)
basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])

# Select entries where the basket price is greater than 0
basket_price = basket_price[basket_price['Basket Price'] > 0]
print(basket_price.sort_values('CustomerID')[:6])




