import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib.pyplot as plt

nrOfTests = 31
test_results = []
smiles = []

for i in range(nrOfTests):
    data = pd.read_excel('/Volumes/SSD/smile_detection/test_results/testperson ' + str(i+1) + '/finaliteration.xlsx')
    test_results.append(data)
    smiles.append(data['Smile:'].tolist())

time = test_results[0]['Seconds: '].tolist()

