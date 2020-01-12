
import matplotlib.pyplot as plt

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
        
        