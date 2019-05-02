import pandas as pd # for data file loading and seeing in tabular form
import matplotlib.pyplot as plt # for plotting
import numpy as np # for array manipulations
import re # for regular expressions.
import sklearn #scikit learn machine learning library for PCA in this project.
from sklearn.decomposition import PCA # Principal component analysis ------> dimensionality reduction for 3-D visualization.
from sklearn.preprocessing import StandardScaler # for data standardization


#Loading the Adult dataset using pandas, a powerful python library for data manipulation
dataset = pd.read_csv("adult.data.txt", names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],sep=',',na_values="?")
print(dataset.tail()) # print last 5 elements of adult data set 



# # Applying PCA to the dataset to visualize it in 3-D while retaining 95% of the original dataset's variance
# from sklearn.preprocessing import StandardScaler
# features = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
#         "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
#         "Hours per week", "Country"] # 14 attributes of the original dataset.  
# # Separating out the features
# x = dataset.loc[:, features].values
# # Separating out the target
# y = dataset.loc[:,['target']].values
# # Standardizing the features
# x = StandardScaler().fit_transform(x)
# print('x: ', x)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=3) # For 3-D visualization of the dataset
# principalComponents = pca.fit_transform(x)
# principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

# finalDf = pd.concat([principalDf, dataset[['target']]], axis = 1)




# Laplace function implementation, takes epsilon as an argument
def Laplacian_func(eps):     
 x=0.01
 mu=0     # mean of laplace distribution
 return ((eps/2.0) * np.exp(-abs(x - mu)*eps)) # laplace distribution function
 
datacount = dataset["Country"].value_counts()    #Store actual data count
tmp = [] # temp list 
mselist = [] # to store mse
fig=plt.figure()


# Call laplace for all values of epsilon, calculate MSE for each case and plot.
noise = Laplacian_func(0.1)     # values from 0.1 to 1.0
noisydata = datacount + noise  # noisy data where noise is laplace distributed
mse = ((datacount- noisydata)**2).mean(axis=0)   # Calculate MSE and store in a list for later plotting
mselist.append(mse) 
noise = Laplacian_func(0.2)
noisydata = datacount + noise
mse = ((datacount-noisydata)**2).mean(axis=0) 
mselist.append(mse)
noise = Laplacian_func(0.3)
noisydata = datacount + noise
mse = ((datacount-noisydata)**2).mean(axis=0)  
mselist.append(mse)
noise = Laplacian_func(0.4)
noisydata = datacount + noise
mse = ((datacount-noisydata)**2).mean(axis=0) 
mselist.append(mse)
noise = Laplacian_func(0.5)
noisydata = datacount + noise
mse = ((datacount-noisydata)**2).mean(axis=0)
mselist.append(mse)
noise = Laplacian_func(0.6)
noisydata = datacount + noise
mse = ((datacount-noisydata)**2).mean(axis=0)
mselist.append(mse)
noise = Laplacian_func(0.7)
noisydata = datacount + noise
mse = ((datacount-noisydata)**2).mean(axis=0)
mselist.append(mse)
noise = Laplacian_func(0.8)
noisydata = datacount + noise
mse = ((datacount-noisydata)**2).mean(axis=0)
mselist.append(mse)
noise = Laplacian_func(0.9)
noisydata = datacount + noise
mse = ((datacount-noisydata)**2).mean(axis=0)
mselist.append(mse)
noise = Laplacian_func(1.0)
noisydata = datacount + noise
mse = ((datacount-noisydata)**2).mean(axis=0)
mselist.append(mse)     # Final list consisting of all the MSE's

for i in range(50):
  for j in list(mselist):
      tmp.append(j)      
print ("Average Mean Square Error is: ", np.average(tmp)) 


epsval=[1.0] 
x=1.0
for i in range(1,10):
   x -= 0.1
   epsval.append(x)   
ax=fig.add_subplot(111)
ax.plot(epsval,mselist)
plt.xlabel('Epsilon')
plt.ylabel('Mean Square Error')
plt.show()



#Exponential Mechanism (Implemented in separate .py file)

# import pandas as pd
# import numpy as np

# # Load the Adult dataset
# dataset = pd.read_csv("adult.data.txt", names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
#         "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
#         "Hours per week", "Country", "Target"],sep=r'\s*,\s*',na_values="?")
# dataset.tail()

# datacount = dataset["Country"].value_counts()

# # Generate random noise from exponential function.
# Exponential_noise = np.random.exponential(1)     # Keep max limit = 1

# print ("Exponentially generated noise:", Exponential_noise)

# """Add random noise drawn from Exponential function to Original data count"""
# noisydata = datacount + Exponential_noise

# #Plot histogram for Noisy data
# noisydata.plot(kind="bar", color = 'r')

# # Laplace Mechanism implemented in separate .py file
# import pandas as pd
# import numpy as np
# import re

# # Load Adult dataset 
# dataset = pd.read_csv("adult.data.txt",
#     names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
#         "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
#         "Hours per week", "Country", "Target"],sep=",",na_values="?")
# dataset.tail()

# # Set parameters for Laplace function implementation
# location = 1.0
# scale = 1.0

# #Find actual data count
# datacount = dataset["Country"].value_counts()

# # Gets random laplacian noise for all values
# Laplacian_noise = np.random.laplace(location,scale, len(datacount))
# print(Laplacian_noise)

# # Add random noise generated from Laplace function to actual count
# noisydata = datacount + Laplacian_noise

# # Generate noisy histogram
# noisydata.plot(kind="bar",color = 'g')

