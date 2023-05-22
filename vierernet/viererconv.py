import torch
import numpy as np
import os
import sys

from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.nn import Conv2d


class ViererNet(torch.nn.Module):
    
    def __init__(self,sizes=[32,32,32], invariant = False,input_dim=1):
        super().__init__()
        lst = [ViererConv(input_dim, sizes[0], 3)] + [ViererConv(sizes[k],sizes[k+1],3) for k in range(len(sizes)-1)] + [ViererConv(sizes[-1],11,3)]
        self.input_dim = input_dim
        self.vierers = torch.nn.ModuleList(lst)
        
        self.tanh = torch.nn.Tanh()
    
        
        
        if invariant:
            self.selector = torch.zeros(1,4,44,dtype=torch.float)    
        else:
            self.selector = torch.nn.Parameter(torch.zeros((1,4,11),dtype=torch.float))
        with torch.no_grad():
            self.selector.data = torch.rand((1,4,11),dtype=torch.float)
            self.selector.data = self.selector.data/self.selector.data.sum(-1,keepdim=True)
            #self.selector[:,0,:]=1.0
        
        self.bn = torch.nn.BatchNorm1d(11)
    def to(self,device):
        super().to(device)
        for vierer in self.vierers:
            vierer.to(device)
        self.selector = self.selector.to(device)
    def forward(self,x):
        
        # x incoming is of size b, k,k
        # need to convert it to size b, g, i, k, k 
        x = x.reshape(-1,1,self.input_dim,x.shape[-1],x.shape[-1])
        v = torch.zeros_like(x)
        
        v = v.expand(-1,3,-1,-1,-1)
        
        x = torch.cat((x,v),dim=1)
        
        #selector = 0
        
        ## viererlayers
        
        for (k,vierer) in enumerate(self.vierers):
            x = vierer(x) # b,g,s,n,n
            #if k<len(self.vierers)-1:
                #selector += self.selector_layersone[k](self.relu(x.mean((-2,-1)).flatten(-2,-1))) # send means into selector net
    
            x = self.tanh(x)
        
        x = x.mean((-1,-2)) # b,4,11
        
        
        
        #selector = self.selector_layerstwo(selector).reshape(-1,4,11) # b,4,11
        #selector = torch.nn.functional.softmax(selector,dim=1)
        selector =self.selector
        logits = (x*selector).sum(1)
    
        
        logits = self.bn(logits)
        return torch.nn.functional.softmax(logits,dim=-1)
    
class SimpleNet(torch.nn.Module):
    def __init__(self,sizes=[32,32,32],input_dim=1):
        super().__init__()
        lst = [Conv2d(input_dim, sizes[0], 3, padding=1)] + [Conv2d(sizes[k],sizes[k+1],3, padding=1) for k in range(len(sizes)-1)] + [Conv2d(sizes[-1],11,3, padding=1)]
        
        self.convlayers = torch.nn.ModuleList(lst)
        self.tanh = torch.nn.Tanh()
        
        self.bn = torch.nn.BatchNorm1d(11)
    
    def forward(self,x):
        
        for lay in self.convlayers:
            x = lay(x)
            x = self.tanh(x)
        x = x.mean((-1,-2))
        x = self.bn(x)
        return torch.nn.functional.softmax(x,dim=-1)

class ViererConv(torch.nn.Module):
    
    def __init__(self,in_channels,out_channels, k):
        super().__init__()
        if not k==3:
            raise NotImplementedError("Only k==3 supported")
            
        self.params = torch.nn.Parameter((1/np.sqrt(in_channels*out_channels)*torch.randn(k**2,out_channels,in_channels)))
        self.k = k
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.get_embedder()
        
        
    def get_embedder(self):
        if not self.k==3:
            raise NotImplementedError("Only k==3 supported")
        
        
        ### we want to build a tensor T of size  4 ,3,3,9
        ## This induces a map from 9 to 4,3,3
        ## T[epsilon,:,:,: ] @ param is the filter \phi_{\epsilon} given by indices param  of size 9
        
        i1 = []
        i2 = []
        i3 = []
        i4 = []
        v = []
        
        #########   epsilon = 0   #########
        
        # k=0
        # 000 0+0 000
        i1.append(0)
        i2.append(1)
        i3.append(1)
        i4.append(0)
        v.append(1)
        
        # k=1
        # 000 +0+ 000
        i1.append(0)
        i2.append(1)
        i3.append(0)
        i4.append(1)
        v.append(1)
    
        i1.append(0)
        i2.append(1)
        i3.append(2)
        i4.append(1)
        v.append(1)
        
        # k=2
        # 0+0 000 0+0
        i1.append(0)
        i2.append(0)
        i3.append(1)
        i4.append(2)
        v.append(1)
        
        i1.append(0)
        i2.append(2)
        i3.append(1)
        i4.append(2)
        v.append(1)
        
        # k=3
        # +0+ 000 +0+
        i1.append(0)
        i2.append(0)
        i3.append(0)
        i4.append(3)
        v.append(1)
        
        i1.append(0)
        i2.append(2)
        i3.append(2)
        i4.append(3)
        v.append(1)
        
        i1.append(0)
        i2.append(0)
        i3.append(2)
        i4.append(3)
        v.append(1)
        
        i1.append(0)
        i2.append(2)
        i3.append(0)
        i4.append(3)
        v.append(1)
        
        
        #########   epsilon = 1   #########

        # k=4
        # 000 -0+ 000
        i1.append(1)
        i2.append(1)
        i3.append(0)
        i4.append(4)
        v.append(-1)
        
        i1.append(1)
        i2.append(1)
        i3.append(2)
        i4.append(4)
        v.append(1)
        
        # k=5 
        # -0+ 000 -0+
        i1.append(1)
        i2.append(0)
        i3.append(0)
        i4.append(5)
        v.append(-1)
        
        i1.append(1)
        i2.append(2)
        i3.append(2)
        i4.append(5)
        v.append(1)
        
        i1.append(1)
        i2.append(0)
        i3.append(2)
        i4.append(5)
        v.append(1)
        
        i1.append(1)
        i2.append(2)
        i3.append(0)
        i4.append(5)
        v.append(-1)
            
      
        #########   epsilon = 2   #########
       
        # k=6
        # 0-0 000 0+0
        i1.append(2)
        i2.append(0)
        i3.append(1)
        i4.append(6)
        v.append(-1)
        
        i1.append(2)
        i2.append(2)
        i3.append(1)
        i4.append(6)
        v.append(1)
        
        # k=7
        # -0- 000 +0+
        i1.append(2)
        i2.append(0)
        i3.append(0)
        i4.append(7)
        v.append(-1)
        
        i1.append(2)
        i2.append(2)
        i3.append(2)
        i4.append(7)
        v.append(1)
        
        i1.append(2)
        i2.append(0)
        i3.append(2)
        i4.append(7)
        v.append(-1)
        
        i1.append(2)
        i2.append(2)
        i3.append(0)
        i4.append(7)
        v.append(1)
            
        ### epsilon = 3
        # k=8
        # -0+ 000 +0-
        i1.append(3)
        i2.append(0)
        i3.append(0)
        i4.append(8)
        v.append(-1)
        
        i1.append(3)
        i2.append(2)
        i3.append(2)
        i4.append(8)
        v.append(-1)
        
        i1.append(3)
        i2.append(0)
        i3.append(2)
        i4.append(8)
        v.append(1)
        
        i1.append(3)
        i2.append(2)
        i3.append(0)
        i4.append(8)
        v.append(1)
            
        
        i1 = torch.tensor(i1,dtype = torch.long)
        i2 = torch.tensor(i2,dtype = torch.long)
        i3 = torch.tensor(i3,dtype = torch.long)
        i4 = torch.tensor(i4,dtype = torch.long)
        
        
        I = i3+3*i2 + 3*3*i1 
                
        
        self.emb = torch.sparse_coo_tensor(torch.stack((I,i4)),torch.tensor(v,dtype=torch.float))
        
        self.mixer= []
        
        # epsilon = 0
        self.mixer.append([[0,0],[1,1],[2,2],[3,3]])
        # epsilon = 1
        self.mixer.append([[0,1],[1,0],[3,2],[2,3]])
        # epsilon = 2
        self.mixer.append([[0,2],[2,0],[1,3],[3,1]])
        # epsilon = 3
        self.mixer.append([[0,3],[3,0],[2,1],[1,2]])

    def to(self,device):
        super().to(device)
        self.emb = self.emb.to(device)
        
        
    def forward(self,x):
        
        # x is of shape batch,g,i,n,n
        n = x.shape[-1]
        
        
        weight = self.emb @ self.params.flatten(-2,-1) # is now of size g*k*k,o*i
        
        weight = weight.reshape(4,self.k,self.k,self.out_channels,self.in_channels) # g,k,k,o,i
        weight = weight.permute(0,3,4,1,2) # g,o,i,k,k
        
        weight = weight.reshape(4*self.out_channels,self.in_channels,self.k,self.k)
        
        outchannels = []
        
        for k in range(4):
            outchannels.append(torch.nn.functional.conv2d(x[:,k,:,:,:],weight,padding=(1,1)).reshape(-1,4,self.out_channels,n,n))
     
        #outchannel[gamma][:,epsilon,:,:,:] is contribution for out[gamma*epsilon]
        
        output = torch.zeros_like(outchannels[0]) # sixe b,g,o,n,n
        
        # mix channels
        for (k,pairs) in enumerate(self.mixer):
            for pair in pairs:
                output[:,k,:,:,:]+= outchannels[pair[0]][:,pair[1],:,:,:]
        
       
        return output
        
class FlipMNIST(MNIST):
    
    def __init__(self, train, download=True, root = 'MNIST'):

        super().__init__(root=root,train=train,download=download,transform= transforms.ToTensor())

        
    def __getitem__(self, index):
        
        
        # load an image
        img,label  = super().__getitem__(index)
        
        # save the original label for analysis purposes
        orglabel = label
        
        # normalize
        img = img - img.mean()
        
        # randomly flip the image and change the class
        theta = torch.rand(1)
        
        if theta<.33:
            return img, torch.tensor(label,dtype=torch.long),torch.tensor(orglabel,dtype=torch.long) 
        elif theta<.66:
            
            if label in [3,4,5,8,9]:
                label = 10
                
            return torch.flip(img,dims=(1,)), torch.tensor(label,dtype=torch.long),torch.tensor(orglabel,dtype=torch.long) 
        else:
            if label in [6,7,8,9]:
                label = 10
                
            return torch.flip(img,dims=(2,)), torch.tensor(label,dtype=torch.long), torch.tensor(orglabel,dtype=torch.long) 

class FlipCIFAR10(CIFAR10):
    
    def __init__(self, train, download=True, root ='CIFAR10',hard=True):
        super().__init__(root=root,train=train,download=download,transform= transforms.ToTensor())
        # hard = True -> double flip dataset
        self.hard = hard
    
    def __getitem__(self, index):
        
        # load an image
        img,label  = super().__getitem__(index)
        
        # save the original label for analysis purposes
        orglabel = label
        
        # normalize
        img = img - img.mean()
        
        # randomly flip the image and change the class
        theta = torch.rand(1)
        
        if theta<.33:
            return img, torch.tensor(label,dtype=torch.long),torch.tensor(orglabel,dtype=torch.long) 
        elif theta<.66:
            
            if label in [3,4,5,8,9]:
                label = 10
                
            return torch.flip(img,dims=(1,)), torch.tensor(label,dtype=torch.long),torch.tensor(orglabel,dtype=torch.long) 
        else:
            if self.hard and label in [6,7,8,9]: 
                label = 10
                
            return torch.flip(img,dims=(2,)), torch.tensor(label,dtype=torch.long), torch.tensor(orglabel,dtype=torch.long) 
    
        
def train_epoch(model,optimizer,loader, weighted = True, data='mnist'):
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    if weighted:
        if data == 'mnist':
            w = torch.tensor([1,1,1,1.5,1.5,1.5,1.5,1.5,3,3,1],device=device,dtype=torch.float)
        else:
            w = torch.tensor([1,1,1,1.5,1.5,1.5,1,1,1.5,1.5,1],device=device,dtype=torch.float)
        loss_fn = torch.nn.CrossEntropyLoss(w)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    
    
    """
    
        array for saving the results.
    
        Layout : res[k] is a 2d-array with results for the images with original label k. 
        res[k][i][j] are the number of examples with the actual label i (i.e. k or 'NaN') that are classified as j (k,'NaN' or something else)
    
    """
    
    res = torch.zeros(10,2,3,device = device)
    
    
    # training epoch
    for (x,label,orglabel) in loader:
        x=x.to(device)
        label=label.to(device)
        orglabel=orglabel.to(device)
    
        pred = net(x) 

        loss = loss_fn(pred,label)
        loss.backward()
        
        #   record classification
        _, predd = torch.max(pred,dim=1)
        
        for k in range(10):
            lab = (orglabel==k)*1.0
            res[k,0,0] += (lab*(label==k)*(predd==k)).sum()
            res[k,0,1] += (lab*(label==k)*(predd==10)).sum()
            res[k,0,2] += (lab*(label==k)).sum()
            
            res[k,1,0] += (lab*(label==10)*(predd==k)).sum()
            res[k,1,1] += (lab*(label==10)*(predd==10)).sum()
            res[k,1,2] += (lab*(label==10)).sum()
            
            
        
        
        optimizer.step()
        optimizer.zero_grad()
    res[:,:,2] = res[:,:,2]-res[:,:,0]-res[:,:,1]
    return model,  res.detach().cpu().numpy()

def test_model(model,testloader):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    model.eval()
    res = torch.zeros(10,2,3,device=device)
    
    # go through test set and save results in the same manner as in train_epoch
    for (x,label,orglabel) in testloader:
       x=x.to(device)
       label=label.to(device)
       orglabel=orglabel.to(device)
   
       pred = net(x) 
       
       _, predd = torch.max(pred,dim=1)

       for k in range(10):
           lab = (orglabel==k)*1.0
           res[k,0,0] += (lab*(label==k)*(predd==k)).sum()
           res[k,0,1] += (lab*(label==k)*(predd==10)).sum()
           res[k,0,2] += (lab*(label==k)).sum()
           
           res[k,1,0] += (lab*(label==10)*(predd==k)).sum()
           res[k,1,1] += (lab*(label==10)*(predd==10)).sum()
           res[k,1,2] += (lab*(label==10)).sum()
    res[:,:,2] = res[:,:,2]-res[:,:,0]-res[:,:,1]       
    return res.detach().cpu().numpy()

def write_result(file, res):
    np.save(file+'.npy',res)
    

if __name__ == "__main__":
    
    nbr_epochs = int(sys.argv[1])
    nbr_runs = int(sys.argv[2])
    dataset = sys.argv[3]
    try:
        result_directory = sys.argv[4]
    except(IndexError):
        result_directory = 'res_'+sys.argv[3]
        
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    if not os.path.exists(result_directory):
        os.mkdir(result_directory)

        
    for typ in ['simple','vierer']:
        for run in range(nbr_runs):
            name = typ+str(run)
        
            # load dataset
            if dataset == 'mnist':
                traindata = FlipMNIST(root= 'MNIST', train=True,download=True)
                testdata = FlipMNIST(root = 'MNIST', train=False,download=True)    
                
                batch_size = 32
                layer_size= 32
                lr = 1e-4
                input_dim=1
                
            elif 'cifar' in dataset:                 
                if 'both' in dataset:
                    traindata = FlipCIFAR10(root='CIFAR', train=True,download=False, dshard = True)
                    testdata = FlipCIFAR10(root='CIFAR', train=False,download=False, hard= True)
                elif 'single' in dataset:
                    traindata = FlipCIFAR10(root='CIFAR',train=True,download=False,hard = False)
                    testdata = FlipCIFAR10(root='CIFAR', train=False,download=False, hard= False)
                else:
                    print('unknown data. Aborting.')
                    quit()
                    
                batch_size=32
                layer_size=64
                lr = 1e-4
                input_dim=3
            
            else:
                print('unknown data. Aborting.')
                quit()
                    
            if 'simple' in typ:
                net = SimpleNet([layer_size,layer_size,layer_size],input_dim=input_dim)
            else:
                net = ViererNet([layer_size,layer_size,layer_size],input_dim=input_dim)
            print('Running experiment '+ str(run) + ' with ' + typ)
            net.to(device)
        
            loader = torch.utils.data.DataLoader(traindata,batch_size=batch_size,num_workers=0)
            testloader = torch.utils.data.DataLoader(testdata,batch_size=batch_size,num_workers=0)
            optimizer = torch.optim.Adam(net.parameters(),lr=lr)
        
            for epoch in range(nbr_epochs):

                net,res = train_epoch(net,optimizer,loader,data=dataset) 
                write_result(os.path.join(result_directory,'train'+name+'_'+str(epoch)),res)
                
                res= test_model(net,testloader)
                
                write_result(os.path.join(result_directory,'test'+name+'_'+str(epoch)),res)

   