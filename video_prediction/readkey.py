#read csv
import csv
import pandas as pd
import h5py  
PATH = 'natops/data/tracking_results2/'

def read_csv(subject, gesture):
    '''
    input: index of subject ,index of gesture,frames(start, end)
    output: numpy array arr[18]
    '''
    '''
    keypoints = []
    with open(PATH+'g%02ds%02d.csv'%(gesture,subject)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:

            keypoints[]
    '''
    columns = [2,53,54,56,57,59,60,62,63,65,66,68,69,71,72,74,75,77,78]
    df=pd.read_csv(PATH+'g%02ds%02d.csv'%(gesture,subject), index_col = 0, sep=',',usecols=columns,skiprows= [0],header=None)       
    return df        

store = pd.HDFStore('keypoints.h5')


for ges in range(24):
    for sub in range(20):
        store['g%02ds%02d'%(ges+1,sub+1)] = read_csv(sub+1, ges+1)  

