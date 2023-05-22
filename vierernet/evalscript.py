import os
import sys
import numpy as np
import matplotlib.pyplot as plt


names = ['simple','vierer']

directory = sys.argv[3]
try:
    experiment_name = sys.argv[4]
except(IndexError):
    experiment_name = sys.argv[3]
    

styles = ['solid','dashed']

rounds = []
for r in range(int(sys.argv[2])):  
    rounds.append(str(r))

perf = [2,len(rounds)]

epochs = int(sys.argv[1])

colors=['blue','red']

totres = np.zeros((len(rounds),epochs, 10,2,3))
fres = np.zeros_like(totres)

for typ in ['train','test']:
    for l,name in enumerate(names):   
        for round in rounds:
            for epoch in range(epochs):
                # load results
                res = np.load(os.path.join(directory,typ+name+str(round)+'_'+str(epoch)+'.npy'))
                totres[int(round), epoch] = res
                
        # normalize to probabilities
        fres[:,:,:,l,:] = totres[:,:,:,l,:]/totres[:,:,:,l,:].sum(3,keepdims=True)
        
        # get accuracy
        acc = (totres[:,:,:,0,0] + totres[:,:,:,1,1]).sum(-1)/totres[:,:,:].sum((2,3,4))


        # determine best performing epoch, and print out median
        print("==========")
        print(typ+' '+name)
        k = np.argmax(np.median(acc,0))
        print(np.median(acc[:,k]))

        perf[l]=acc[:,k]
         
        # plot with errorbar
        plt.errorbar(np.arange(epochs),np.median(acc,0),yerr=np.vstack((np.percentile(acc,87.5,axis=0)-np.median(acc,0), np.median(acc,0) -np.percentile(acc,12.5,axis=0))),color=colors[l], linestyle=styles[l],elinewidth=.5) 
        plt.title(experiment_name + ' (' + typ+')')
        
    # perform a non parametric test to determine p-value for "vierernet better than baseline"
    # if close to 1, baseline is better, with p-value (1-displayed number)
    print('p(vierer>baseline)')
    print((np.reshape(perf[0],(len(rounds),1))>np.reshape(perf[1],(1,len(rounds)))).sum()/(len(rounds)**2))
    
    # intervals that made nice figures for experiments in paper.
    # may need to be modified. If dataset name is not specified, automatic limits will be used
    if experiment_name == 'mnist':
        plt.ylim([.6,1])
    elif experiment_name == 'cifar_single':
        plt.ylim([.25,.65])
    elif experiment_name == 'cifar_both':
        plt.ylim([.25,.5])
    plt.legend(['baseline','vierer'],loc=4)
    
    if not os.path.exists('plots'):
        os.mkdir('plots')
    plt.savefig(os.path.join('plots',directory + '_'+typ),dpi=150)
    plt.figure()
    
    
    


