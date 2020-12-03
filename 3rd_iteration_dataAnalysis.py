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

for i in range(nrOfTests):
    data = pd.read_excel('/Volumes/SSD/smile_detection/test_results/testperson ' + str(i + 1) + '/finaliteration.xlsx')
    test_results.append(data)
    smile.append(data['Smile:'].tolist())

smileCount = []
time = test_results[0]['Seconds: '].tolist()

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
time = list(means_of_slices(time, 40))

print(len(smile))

surveyData = pd.read_csv('survey.csv', names=['Alder', 'Kon', 'Hvordan vil du bedømme det klip du lige har set?','Har du set klippet før?', 'Har du set noget fra tv-serien før?'])

surveyData = StandardScaler().fit_transform(surveyData)

print(surveyData)
plt.bar(*(time,smile[0]), width=10.0, align='edge')
plt.show()

""""

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
"""