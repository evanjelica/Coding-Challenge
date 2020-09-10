#ACM Research Coding Challenge

#Imported packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

#Loading dataset 
data = pd.read_csv('ClusterPlot.csv')
data.head()

#Scaling the dataset
mms = MinMaxScaler()
mms.fit(data)
data_transformed = mms.transform(data)

#Elbow Method
Sum_of_squared_distances = []
K = range (1,6)
for k in K:
    km = KMeans (n_clusters = k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow method for Optimal K')
plt.show()