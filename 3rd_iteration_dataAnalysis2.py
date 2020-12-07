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
import xlsxwriter

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

smile = np.array(smile)
for i in range(len(smile)):
    smile[i] = smile[i].flatten()

workbook = xlsxwriter.Workbook('smiles.xlsx')
worksheet = workbook.add_worksheet()

worksheet.write('A1', '0-10')
worksheet.write('B1', '10-20')
worksheet.write('C1', '20-30')
worksheet.write('D1', '30-40')
worksheet.write('E1', '40-50')
worksheet.write('F1', '50-60')
worksheet.write('G1', '60-70')
worksheet.write('H1', '70-80')
worksheet.write('I1', '80-90')
worksheet.write('J1', '90-100')
worksheet.write('K1', '100-110')
worksheet.write('L1', '110-120')
worksheet.write('M1', '120-130')
worksheet.write('N1', '130-140')
worksheet.write('O1', '140-150')
worksheet.write('P1', '150-160')
worksheet.write('Q1', '160-170')
worksheet.write('R1', '170-180')
worksheet.write('S1', '180-190')
worksheet.write('T1', '190-200')
worksheet.write('U1', '200-210')
worksheet.write('V1', '210-220')
worksheet.write('W1', '220-230')
worksheet.write('X1', '230-240')
worksheet.write('Y1', '240-250')
worksheet.write('Z1', '250-260')
worksheet.write('AA1', '260-270')
worksheet.write('AB1', '270-280')
worksheet.write('AC1', '280-290')
worksheet.write('AD1', '290-300')

for i in range(len(smile)):
    for j in range(len(smile[i])):
        worksheet.write('A'+str(i+2), '290-300')


#prints the amount of "smiles" lists it has stored?
print(smile)


#Beginning of PCA data visualisation.
"""""
df = px.data.iris() //Loads iris data from a build in function
features = ["sepal_width", "sepal_length", "petal_width", "petal_length"]

fig = px.scatter_matrix(
    df,
    dimensions=features,
    color="species"
)


fig.update_traces(diagonal_visible=False)
fig.show()


df = px.data.NummereretArk1()
X = df[['Age', 'Rating', 'Watched_clip', 'Watched_series']]

pca = PCA(n_components=2)
components = pca.fit_transform(X)

fig = px.scatter(components, x=0, y=1, color=df['Gender'])
fig.show()
"""""

df = pd.read_csv('Copy of smiles, seasons.csv', names=['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100','100-110','110-120','120-130','130-140','140-150','150-160','160-170','170-180','180-190','190-200','200-210','210-220','220-230','230-240','240-250','250-260','260-270','270-280','280-290','290-300','Target'])
print(df)
df.head()
display(df)

#Opdel i to lister, en for target (som her hedder Gender) og en for features:
features = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100','100-110','110-120','120-130','130-140','140-150','150-160','160-170','170-180','180-190','190-200','200-210','210-220','220-230','230-240','240-250','250-260','260-270','270-280','280-290','290-300']
x = df.loc[:, features].values

y = df.loc[:, ['Target']].values

#omregnes så det kan bruges til PCA. Dette er scaling to unit variance.
x = StandardScaler().fit_transform(x)

pd.DataFrame(data=x, columns=features).head()

# Printer den standardiserede matrice over spørgeskema værdierne.
print(x)
print(y)


#PCA!

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf.head(5)
df[['Target']].head()

finalDf = pd.concat([principalDf, df[['Target']]], axis = 1)
finalDf.head(5)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)


targets = ['Jeg har set flere sæsoner', 'Jeg har set flere afsnit', 'Jeg har set et enkelt afsnit','Nej, jeg har aldrig set det før']
colors = ['r', 'g', 'b','m']
for Target, color in zip(targets,colors):
    indicesToKeep = finalDf['Target'] == Target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
ax.legend(targets)
ax.grid()

#plt.tight_layout()
plt.show()


# prints out the smiles for participant 0.
#plt.bar(*(time,smile[0]), width=10.0, align='edge')
#plt.show()

"""""
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