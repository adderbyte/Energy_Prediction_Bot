
import matplotlib.pyplot as plt
import os 
import pandas as pd
import tensorflow as tf

def plot_meter(train, leak, start=0, n=200, bn=10):
    '''
    plot data for building meter readingss
    
    '''
    assert bn < leak.building_id.nunique();
    for bid in leak.building_id.unique()[:bn]:    
        tr = train[train.building_id == bid]
        lk = leak[leak.building_id == bid]
        
        #for m in lk.meter.unique():
        #    plt.figure(figsize=[10,2])
        #    trm = tr[tr.meter == m]
        #    lkm = lk[lk.meter == m]
            
        plt.plot(tr.timestamp[start:start+n], tr.meter_reading.values[start:start+n], label='true values')    
        plt.plot(lk.timestamp[start:start+n], lk.meter_reading.values[start:start+n], '--', label='predicted values')
        plt.title('bid:{}, meter:{}'.format(bid, 0))
        plt.legend()
        plt.show()
        
   

def tfdatabuilder(cleanedpath,cleandataName,trainName, validateName, validation_split,test=False ):
    
    # get data set
    if cleandataName.endswith('feather'):
        train_test = pd.read_feather(cleanedpath +  cleandataName )
    else:
        train_test = pd.read_csv(cleanedpath +  cleandataName )
    
    #print(train_test.columns)
    # set train index
    train_test.set_index('timestamp',inplace=True);
    # get column names
    
    if test: del train_test['index']
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
    
    if not test:
    
        # separate train and validation
        test = train_test.tail(validation_split).copy(deep=True)
        train_= train_test.shape[0]-test.shape[0] # train partition
        train = train_test[:train_].copy(deep=True)

        # savedata
        train.to_csv(cleanedpath + trainName,index=False)
        test.to_csv(cleanedpath + validateName,index=False)
        print('==========***train option**=========================')
        print('Done')

        return train.index.values,test.index.values,data_types, col_names
    else:
        
        train_test.to_csv(cleanedpath + trainName,index=False)
        print('=====***test option ***================')
        print('Done')
        return data_types, col_names
  
def tfpreprocess(train):
    
    data_types = list(train.dtypes)#[1:]
    # convert the data types to 
    # tensorflow data types 
    for i,j in enumerate(data_types):
        j = str(j) # convertr from numpy dtype to str
        #print(j)
        if j == 'float64':
            data_types[i] = tf.float64
        else:
            data_types[i]  = tf.int32
    col_names = list(train.columns)#[1:]
    return data_types,col_names
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        