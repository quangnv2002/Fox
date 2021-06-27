import numpy as np
import pandas as pd
from pandas import plotting

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

import plotly.offline as py
from plotly.offline import init_notebook_mode
from plotly.offline import iplot

import plotly.graph_objs as go
from plotly import tools
init_notebook_mode( connected = True )
import plotly.figure_factory as ff

import os
print(os.listdir('input'))      // LIỆT KÊ DANH SÁCH CÁC FILE TRONG THƯ MỤC "input"

data = pd.read_csv('input/Khách-hàng_data.csv')     // ĐỌC FILE CSV "khách_hàng_data.csv"
dat = ff.create_table(data.head())                  // TẠO BẢNG 
py.iplot(dat)

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (18,8)

plt.subplot(1,2,1)      // chia cửa sổ hiện tại thành một ma trận m x n khoảng để vẽ đồ thị, và chọn p là cửa sổ hoạt động. 
                        // Các đồ thị thành phần được đánh số từ trái qua phải, từ trên xuống dưới, sau đó đến hàng thứ hai

sns.set(style = 'whitegrid')        // SET STYLE CHO SEABORN
sns.distplot(data['Annual Income (k$)'])    // CHỌN CỘT ĐỂ VẼ ?
plt.title('Distribution of Annual Income',fontsize = 20)    // XÉT TIÊU ĐỀ CHO BẢNG
plt.xlabel('Range of Annual Income')    // XÉT TIÊU ĐỀ CHO CỘT Ox
plt.ylabel('Count')                     // XÉT TIÊU ĐỀ CHO CỘT Oy

plt.subplot(1,2,2)
sns.set(style = 'whitegrid')
sns.distplot(data['Age'],color = 'red')
plt.title('Distribution of Age', fontsize = 20)
plt.xlabel('Range of Age')
plt.ylabel('Count')
plt.show()

labels = ['Female','Male']
size = data['Gender'].value_counts()
colors = ['lightgreen','orange']
explode = [0,0.1]

plt.rcParams['figure.figsize'] = (9,9)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow= True, autopct='%.2f%%')
plt.title('Gender',fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()

x = data.iloc[:, [3,4]].values

print(x.shape)

from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    km = KMeans(n_clusters= i, init = 'k-means++', max_iter= 300, n_init= 10, random_state= 0)
    km.fit(x)
    wcss.append(km.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method',fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()

km = KMeans(n_clusters=5, init='k-means++', max_iter= 300, n_init= 10, random_state= 0)
y_means = km.fit_predict(x)

plt.scatter(x[y_means==0,0], x[y_means==0,1], s=100, c='pink', label = 'mister')

plt.scatter(x[y_means==1,0], x[y_means==1,1], s=100, c='yellow', label = 'general')

plt.scatter(x[y_means==2,0], x[y_means==2,1], s=100, c='cyan', label = 'target')

plt.scatter(x[y_means==3,0], x[y_means==3,1], s=100, c='magenta', label = 'spendthrift')

plt.scatter(x[y_means==4,0], x[y_means==4,1], s=100, c='orange', label = 'careful')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=50, c='blue', label = 'centeroid')

plt.style.use('fivethirtyeight')
plt.title('K Means Clustering',fontsize = 20)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()
