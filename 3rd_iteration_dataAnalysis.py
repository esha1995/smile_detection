import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import pandas as pd
from itertools import islice
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from IPython.display import display
import plotly.express as px

#Definition for inddeling af smil i 10 sek intervaller.
def means_of_slices(iterable, slice_size):
    iterator = iter(iterable)
    while True:
        slice = list(islice(iterator, slice_size))
        if slice:
            yield sum(slice)/len(slice)
        else:
            return


nrOfTests = 31
test_results = []
smile = []

#Læser smil excelværdierne for alle testpersoner og gemmer den i en liste der hedder smile.
for i in range(nrOfTests):
    data = pd.read_excel('test_results/testperson '+str(i + 1)+'/finaliteration.xlsx')
    test_results.append(data)
    smile.append(data['Smile:'].tolist()) #Refers to list in xlsx document with name "Smile: ", so that it only stores this category of values

smileCount = []
time = test_results[0]['Seconds: '].tolist() #Refers to list in xlsx document with name "Seconds: ", so that it only stores this category of values

#Removes invalid smiles from the list.
for i in range(len(smile)):
    smileCount.append(0)
    for j in range(1, len(smile[i])):
        if(smile[i][j] == 1 and smile[i][j-1] == 0 and smile[i][j+1] == 0):
            smile[i][j] = 0
    for j in range(1, len(smile[i])):
        if(smile[i][j-1] == 0 and smile[i][j] == 1):
            smileCount[i] += 1
for i in range(len(smile)):
    smile[i] = list(means_of_slices(smile[i], 40))
time = list(means_of_slices(time, 40)) #40 bruges, da vi har hvert 1/4 af et sekund målt. Det vil til slut give 10 sek. inddelinger.

#prints the amount of "smiles" lists it has stored?
print(len(smile))


#Beginning of PCA data visualisation.
"""""
df = px.data.iris()
features = ["sepal_width", "sepal_length", "petal_width", "petal_length"]

fig = px.scatter_matrix(
    df,
    dimensions=features,
    color="species"
)


fig.update_traces(diagonal_visible=False)
fig.show()
"""""

df = px.data.NummereretArk1()
X = df[['Age', 'Rating', 'Watched_clip', 'Watched_series']]

pca = PCA(n_components=2)
components = pca.fit_transform(X)

fig = px.scatter(components, x=0, y=1, color=df['Gender'])
fig.show()

"""""
#df = pd.read_csv('Nummereret.csv', names=['Age', 'Gender', 'Rating 1-5','Have you seen the clip before?', 'Have you seen anything from the tv-show before?'])

df.head()
display(df)

#Opdel i to lister, en for target (som her hedder Gender) og en for features:
features = ['Age', 'Rating 1-5','Have you seen the clip before?', 'Have you seen anything from the tv-show before?']
x = df.loc[:, features].values

y = df.loc[:, ['Gender']].values

#omregnes så det kan bruges til PCA. Dette er scaling to unit variance.
x = StandardScaler().fit_transform(x)

pd.DataFrame(data = x, columns = features).head()

# Printer den standardiserede matrice over spørgeskema værdierne.
print(x)
print(y)



#PCA!

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf.head(5)
df[['Gender']].head()

finalDf = pd.concat([principalDf, df[['Gender']]], axis = 1)
finalDf.head(5)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)


targets = ['1', '2']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Gender'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
ax.legend(targets)
ax.grid()

#plt.tight_layout()
plt.show()


# prints out the smiles for participant 0.
plt.bar(*(time,smile[10]), width=10.0, align='edge')
plt.show()


#Plots of smiles based on rating. All participants with the same rating has 


all = np.sum(smile,0)
for i in range(len(all)):
    all[i] = (all[i]/len(smile))*100

# all who rated poor
poor = []
poor.append(smile[5])
poor.append(smile[16])
poor.append(smile[17])
poorSum = np.sum(poor,0)
for i in range(len(poorSum)):
    poorSum[i] = (poorSum[i]/len(poor))*100

# all who rated fair
fair = []
fair.append(smile[1])
fair.append(smile[7])
fair.append(smile[9])
fair.append(smile[10])
fair.append(smile[12])
fair.append(smile[15])
fair.append(smile[20])
fair.append(smile[27])
fair.append(smile[28])
fairSum = np.sum(fair,0)
for i in range(len(fairSum)):
    fairSum[i] = (fairSum[i]/len(fair))*100



# all who rated good
good = []
good.append(smile[3])
good.append(smile[13])
good.append(smile[18])
good.append(smile[19])
good.append(smile[23])
goodSum = np.sum(good,0)
for i in range(len(goodSum)):
    goodSum[i] = (goodSum[i]/len(good))*100

# all who rated very good
veryGood = []
veryGood.append(smile[4])
veryGood.append(smile[8])
veryGood.append(smile[11])
veryGood.append(smile[14])
veryGood.append(smile[21])
veryGood.append(smile[24])
veryGood.append(smile[25])
veryGood.append(smile[26])
veryGood.append(smile[29])
veryGood.append(smile[30])
veryGoodSum = np.sum(veryGood,0)
for i in range(len(veryGoodSum)):
    veryGoodSum[i] = (veryGoodSum[i]/len(veryGood))*100


# all who rated very excellent
excellent = []
excellent.append(smile[0])
excellent.append(smile[2])
excellent.append(smile[6])
excellentSum = np.sum(excellent,0)
for i in range(len(excellentSum)):
    excellentSum[i] = (excellentSum[i]/len(excellent))*100

# all who rated poor and fair
poorAndFair = poor + fair
poorAndFairSum = np.sum(poorAndFair,0)
for i in range(len(poorAndFairSum)):
    poorAndFairSum[i] = (poorAndFairSum[i]/len(poorAndFair))*100


# all who rated very good and excellent
veryGoodAndExcellent = veryGood + excellent
veryGoodAndExcellentSum = np.sum(veryGoodAndExcellent,0)
for i in range(len(veryGoodAndExcellentSum)):
    veryGoodAndExcellentSum[i] = (veryGoodAndExcellentSum[i]/len(veryGoodAndExcellent))*100



fig, axs = plt.subplots(4)
axs[0].plot(time,poorAndFairSum)
axs[0].set_title('Poor and Fair ' + str(len(poorAndFair))+'/31')
axs[0].set_ylim(0,100)
axs[0].set_ylabel('0 - 100 % of participants')
axs[1].plot(time,goodSum)
axs[1].set_title('Good ' + str(len(good))+'/31')
axs[1].set_ylim(0,100)
axs[2].plot(time,veryGoodAndExcellentSum)
axs[2].set_title('Very good and excellent ' + str(len(veryGoodAndExcellent))+'/31')
axs[2].set_ylim(0,100)
axs[3].plot(time,all)
axs[3].set_title('All 31/31')
axs[3].set_ylim(0,100)
axs[3].set_xlabel('Duration of clip')
plt.legend()
plt.show()
"""""