
import matplotlib.pyplot as plt
import os 
import pandas as pd
import tensorflow as tf

def plot_meter(train, leak, start=0, n=100, bn=10,n_plots=5):
    '''
    plot data for building meter readingss
    
    '''
    count = 0
    for bid in leak.building_id.unique()[:bn]:    
        tr = train[train.building_id == bid]
        lk = leak[leak.building_id == bid]
        count +=1
        #for m in lk.meter.unique():
        #    plt.figure(figsize=[10,2])
        #    trm = tr[tr.meter == m]
        #    lkm = lk[lk.meter == m]
            
        plt.plot(tr.timestamp[start:start+n], tr.meter_reading.values[start:start+n], label='true values')    
        plt.plot(lk.timestamp[start:start+n], lk.meter_reading.values[start:start+n], '--', label='predicted values')
        plt.title('bid:{}, meter:{}'.format(bid, 0))
        plt.legend()
        plt.show()
        if count > n_plots:
            break
   

def tfdatabuilder(cleanedpath,cleandataName,trainName, validateName, validation_split ):
    
    # get data set
    train_test = pd.read_csv(cleanedpath +  cleandataName )
    cleandataName# set index to timestamp
    train_test.set_index('timestamp',inplace=True);
    # del colum unnmaed
    #del train['Unnamed: 0'];
    # get column names
    #print(train_test.columns)
    col_names = list(train_test.columns)#[1:]
    print(col_names)
    # save the data types 
    data_types = list(train_test.dtypes)#[1:]
    # convert the data types to 
    # tensorflow data types 
    for i,j in enumerate(data_types):
        j = str(j) # convertr from numpy dtype to str
        #print(j)
        if j == 'float64':
            data_types[i] = tf.float64
        else:
            data_types[i]  = tf.int32
    
    print(data_types)
    
    # separate train and validation
    test = train_test.tail(validation_split).copy(deep=True)
    train_= train_test.shape[0]-test.shape[0] # train partition
    train = train_test[:train_].copy(deep=True)
    
    # savedata
    train.to_csv(cleanedpath + trainName,index=False,header=None)
    test.to_csv(cleanedpath + validateName,index=False,header=None)
    print('====================================')
    print('Done')
    
    return train.index.values,test.index.values,data_types, col_names
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        