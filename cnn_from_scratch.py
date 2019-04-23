#------------------------------------------------------------------------------------------------------------------------------------------
#import libraries
from torchvision.datasets import EMNIST
from torch.utils.data import ConcatDataset, Subset
from torchvision.transforms import ToTensor, Compose
from torchvision.datasets import MNIST
import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD # just choose which to use
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
import statistics

#to calculate mean and standard deviation
def cal_mean(lst):
  return sum(lst) / len(lst)
def cal_std(lst):
  return statistics.stdev(lst)

class MyMNIST(Dataset):
  '''
  ds: mnist dataset downloaded from PyTorch
  flatten: bool, flatten it or not
  '''
  def __init__(self, ds, flatten):
    self.flatten = flatten
    self.ds = ds
    
  def __len__(self):
    return len(self.ds)
  
  def __getitem__(self, index):
    image, label = self.ds[index]
    image = image.view(-1) if self.flatten else image
    return image, label
#----------------------------------------------------------------------------------------------------------------------------------------
#train function for validation phase
#arr1 and arr2 used to store loss for each epoch
def train_CNN_valid(model, loaders, optimizer, writer, n_epochs, ckpt_path, device='cpu'):
  arr1 = []
  arr2 = []
  def run_epoch(train_or_eval):
    epoch_loss = 0.
    epoch_acc = 0.
    for i, batch in enumerate(loaders[train_or_eval], 1):
      in_data, labels = batch
      in_data, labels = in_data.to(device), labels.to(device)
      
      if train_or_eval == 'train':
        optimizer.zero_grad()
        
      logits = model(in_data)
      batch_loss = model.loss(logits, labels)
      batch_acc = model.top1_accuracy(logits, labels)
      
      epoch_loss += batch_loss.item()
      epoch_acc += batch_acc
      
      if train_or_eval == 'train':
        batch_loss.backward()
        optimizer.step()
        
    epoch_loss /= i
    epoch_acc /= i
    
    losses[train_or_eval] = epoch_loss
    accs[train_or_eval] = epoch_acc

    if writer is None:
      print('epoch %d %s loss %.4f acc %.4f' % (epoch, train_or_eval, epoch_loss, epoch_acc))
    elif train_or_eval == 'eval':
      writer.add_scalars('%s_loss' % model.__class__.__name__, # CnnClassifier or FcClassifier
                         tag_scalar_dict={ 
                                          'eval': losses['eval']}, 
                         global_step=epoch)
      
      writer.add_scalars('%s_top1_accuracy' % model.__class__.__name__, # CnnClassifier or FcClassifier
                         tag_scalar_dict={'train': accs['train'], 
                                          'eval': accs['eval']}, 
                         global_step=epoch)
      
      # For instructional purpose, add images here, just the last in_data
      if epoch % 10 == 0:
        if len(in_data.size()) == 2: # when it is flattened, reshape it
          in_data = in_data.view(-1, 1, 28, 28)
          
        img_grid = make_grid(in_data.to('cpu'))
        writer.add_image('%s/eval_input' % model.__class__.__name__, img_grid, epoch)
    return epoch_loss    
  # main statements
  losses = dict()
  accs = dict()
  
  for epoch in range(1, n_epochs+1):
    x = run_epoch('train')
    y = run_epoch('eval')
    arr1.append(x)
    arr2.append(y)
    # For instructional purpose, show how to save checkpoints
    if ckpt_path is not None:
      torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'losses': losses,
        'accs': accs
      }, '%s/%d.pt' % (ckpt_path, epoch))
  arr1.sort()
  arr2.sort()
  return arr1[0], arr2[0]

#train function for testing phase
#arr1,arr2,arr3,arr4,arr5,arr6 used to store metrics: loss, top 1 accuracy and top 3 accuracy for training and testing phase
#arr1,arr2,arr3 used to store values for training phase and arr4,arr5,arr6 used to store values for testing phase
def train_CNN(model, loaders, optimizer, writer, n_epochs, ckpt_path, device='cpu'):
  arr1 = []
  arr2 = []
  arr3 = []
  arr4 = []
  arr5 = []
  arr6 = []
  def run_epoch(train_or_eval):
    epoch_loss = 0.
    epoch_acc = 0.
    epoch_acc2 = 0.
    for i, batch in enumerate(loaders[train_or_eval], 1):
      in_data, labels = batch
      in_data, labels = in_data.to(device), labels.to(device)
      
      if train_or_eval == 'train':
        optimizer.zero_grad()
        
      logits = model(in_data)
      batch_loss = model.loss(logits, labels)
      batch_acc = model.top1_accuracy(logits, labels)
      batch_acc2 = float(model.top3_accuracy(logits, labels)[0])
      epoch_loss += batch_loss.item()
      epoch_acc += batch_acc
      epoch_acc2 += batch_acc2
      
      if train_or_eval == 'train':
        batch_loss.backward()
        optimizer.step()
        
    epoch_loss /= i
    epoch_acc /= i
    epoch_acc2 /= i
    
    losses[train_or_eval] = epoch_loss
    accs[train_or_eval] = epoch_acc
    accs2[train_or_eval] = epoch_acc2
    
    if writer is None:
      print('epoch %d %s loss %.4f acc %.4f' % (epoch, train_or_eval, epoch_loss, epoch_acc))
    elif train_or_eval == 'eval':
      writer.add_scalars('%s_loss' % model.__class__.__name__, # CnnClassifier or FcClassifier
                         tag_scalar_dict={'train': losses['train'], 
                                          'eval': losses['eval']}, 
                         global_step=epoch)
      
      writer.add_scalars('%s_top1_accuracy' % model.__class__.__name__, # CnnClassifier or FcClassifier
                         tag_scalar_dict={'train': accs['train'],
                                          'eval': accs['eval']},
                         global_step=epoch)

      writer.add_scalars('%s_top3_accuracy' % model.__class__.__name__, # CnnClassifier or FcClassifier
                         tag_scalar_dict={'train': accs2['train'], 
                                          'eval': accs2['eval']}, 
                         global_step=epoch)
      # For instructional purpose, add images here, just the last in_data
      if epoch % 10 == 0:
        if len(in_data.size()) == 2: # when it is flattened, reshape it
          in_data = in_data.view(-1, 1, 28, 28)
        #if len(logits.size()) == 2: # when it is flattened, reshape it
        #  logits = logits.view(-1, 1, 28, 28)  
        

        img_grid = make_grid(in_data.to('cpu'))
        #out_gtid = make_grid(logits.to('cpu'))
        writer.add_image('%s/eval_input' % model.__class__.__name__, img_grid, epoch)
        #writer.add_image('%s/eval_output' % model.__class__.__name__, out_img_grid, epoch)
    return epoch_loss,epoch_acc,epoch_acc2
  # main statements
  losses = dict()
  accs = dict()
  accs2 = dict()
  for epoch in range(1, n_epochs+1):
    x,y,z = run_epoch('train')
    arr1.append(x)
    arr2.append(y)
    arr3.append(z)
    x,y,z = run_epoch('eval')
    arr4.append(x)
    arr5.append(y)
    arr6.append(z)
    # For instructional purpose, show how to save checkpoints
    if ckpt_path is not None:
      torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'losses': losses,
        'top1accs': accs,
        'top3accs': accs2,
      }, '%s/%d.pt' % (ckpt_path, epoch))
  arr1.sort()
  arr2.sort(reverse=True)
  arr3.sort(reverse=True)
  arr4.sort()
  arr5.sort(reverse=True)
  arr6.sort(reverse=True)
  return arr1[0],arr2[0],arr3[0],arr4[0],arr5[0],arr6[0]


def get_datasets(split='balanced', save=False):
  download_folder = './data'
  
  transform = Compose([ToTensor()])

  dataset = ConcatDataset([EMNIST(root=download_folder, split=split, download=True, train=False, transform=transform),
                           EMNIST(root=download_folder, split=split, download=True, train=True, transform=transform)])
    
  # Ignore the code below with argument 'save'
  if save:
    random_seed = 4211 # do not change
    n_samples = len(dataset)
    eval_size = 0.2
    indices = list(range(n_samples))
    split = int(np.floor(eval_size * n_samples))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices, eval_indices = indices[split:], indices[:split]

    # cut to half
    train_indices = train_indices[:len(train_indices)//2]
    eval_indices = eval_indices[:len(eval_indices)//2]

    np.savez('train_test_split.npz', train=train_indices, test=eval_indices)
  
  # just use save=False for students
  # load train test split indices
  else:
    with np.load('./train_test_split.npz') as f:
      train_indices = f['train']
      eval_indices = f['test']

  train_dataset = Subset(dataset, indices=train_indices)
  eval_dataset = Subset(dataset, indices=eval_indices)
  
  return train_dataset, eval_dataset

#to split the training data set for validation phase
#random state has been set to randomize the split
def splitvalidation(trainingd):
  train_ds,eval_ds = train_test_split(trainingd,test_size=0.2,random_state=42)
  return train_ds, eval_ds

#------------------------------------------------------------------------------------------------------------------------------------------------
#classifier model definition for CNN from scratch
class CnnClassifier(nn.Module):
  def __init__(self, n_hidden):
    super(CnnClassifier, self).__init__()
    
    # in_data size: (batch_size, 1, 28, 28)
    self.cnn_layers = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=0),
      nn.ReLU(),
      nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=0),
      nn.ReLU(),
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=1, padding=0),
      nn.Conv2d(in_channels=16, out_channels=32,kernel_size=3,stride = 1, padding=0),
      nn.Sigmoid()
    )
    # linear layers transforms flattened image features into logits before the softmax layer
    self.linear = nn.Sequential(
      nn.Linear(32, n_hidden),
      nn.ReLU(),
      nn.Linear(n_hidden, 47)
    )
    
    self.softmax = nn.Softmax(dim=1)
    self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
    
  def forward(self, in_data):
    img_features = self.cnn_layers(in_data).view(in_data.size(0), 32)# in_data.size(0) == batch_size
    logits = self.linear(img_features)
    return logits
  
  def loss(self, logits, labels):
    preds = self.softmax(logits) # size (batch_size, 10)
    return self.loss_fn(preds, labels) / logits.size(0) # divided by batch_size
  
  def top1_accuracy(self, logits, labels):
    # get argmax of logits along dim=1 (this is equivalent to argmax of predicted probabilites)
    predicted_labels = torch.argmax(logits, dim=1, keepdim=False) # size (batch_size,)
    n_corrects = predicted_labels.eq(labels).sum(0) # sum up all the correct predictions
    return int(n_corrects) / float(logits.size(0)) * 100. # in percentage
  
  #reference: https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
  def top3_accuracy(self, logits, labels,topk=(3,)):
    maxk = max(topk)
    batch_size = labels.size(0)

    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
#_______________________________________________________________________________________________________________

#main function for validation phase
#execute for 6 times with differnt metrices
def cnn_main_val():
  valid_hidden = [32,64]
  valid_optlr = [('adam',0.001),('sgd',0.1),('sgd',0.01)]
  min_loss = []
  min_loss2 = []
  x = 0
  for a in valid_hidden:
    for b in valid_optlr:
      parser = ArgumentParser()
      parser.add_argument('--hidden', type=int, default=a)
      parser.add_argument('--gpu', type=int, default=-1)
      parser.add_argument('--lr', type=float, default=b[1])
      parser.add_argument('--batch', type=int, default=32)
      parser.add_argument('--ckpt', type=str, default='./ckpt/cnn_val%s' %a+str(b[1]) )
      parser.add_argument('--epoch', type=int, default=10)
      parser.add_argument('--optim', type=str, default=b[0])
      args = parser.parse_args()
      n_hidden = args.hidden
      gpu = args.gpu
      lr = args.lr
      batch_size = args.batch
      ckpt_path = args.ckpt
      n_epochs = args.epoch
      opt_str = args.optim
  
      ckpt_path = '%s/%s' % (ckpt_path, opt_str)
  
      if ckpt_path is not None:
        if not(os.path.exists(ckpt_path)):
          os.makedirs(ckpt_path)

      if gpu == -1:
        DEVICE = 'cpu'
      elif torch.cuda.is_available():
        DEVICE = gpu
      model =  CnnClassifier(n_hidden).to(DEVICE)
      trainu,evalu = get_datasets()
  
      if opt_str == 'adam':
        opt_class = Adam
      elif opt_str == 'sgd':
        opt_class = SGD
      #split training set into train and eval data
      valitrai, valieva = splitvalidation(trainu)

      dataloaders = {
        'train': DataLoader(MyMNIST(valitrai, flatten = False), batch_size=batch_size, drop_last=False,shuffle=True),
        'eval': DataLoader(MyMNIST(valieva, flatten = False), batch_size=batch_size, drop_last=False)
      }
      optimizer = opt_class(model.parameters(), lr=lr)
      writer = SummaryWriter('./logs/cnn/%s' % opt_str+str(n_hidden)+str(lr))
      temp1, temp2 = train_CNN_valid(model, dataloaders, optimizer, writer, n_epochs, ckpt_path, DEVICE)
      min_loss.append(temp1)
      min_loss2.append(temp2)
  return min_loss, min_loss2

#main function for testing phase
#returns the 6 lists which store loss, top1 accuracy and top3 accuracy for both training and testing phase

def cnn_main(in_hidden,in_optimizer, in_lr, number):
      parser = ArgumentParser()
      parser.add_argument('--hidden', type=int, default=in_hidden)
      parser.add_argument('--gpu', type=int, default=-1)
      parser.add_argument('--lr', type=float, default=in_lr)
      parser.add_argument('--batch', type=int, default=32)
      parser.add_argument('--ckpt', type=str, default='./ckpt/cnn/{}'.format(number))
      parser.add_argument('--epoch', type=int, default=50)
      parser.add_argument('--optim', type=str, default=in_optimizer)
      args = parser.parse_args()
      n_hidden = args.hidden
      gpu = args.gpu
      lr = args.lr
      batch_size = args.batch
      ckpt_path = args.ckpt
      n_epochs = args.epoch
      opt_str = args.optim
  
      ckpt_path = '%s/%s' % (ckpt_path, opt_str)
  
      if ckpt_path is not None:
        if not(os.path.exists(ckpt_path)):
          os.makedirs(ckpt_path)

      if gpu == -1:
        DEVICE = 'cpu'
      elif torch.cuda.is_available():
        DEVICE = gpu
      model =  CnnClassifier(n_hidden).to(DEVICE)
      trainu,evalu = get_datasets()
      dataloaders = {
        'train': DataLoader(MyMNIST(trainu, flatten = False), batch_size=batch_size, drop_last=False,shuffle=True),
        'eval': DataLoader(MyMNIST(evalu, flatten = False), batch_size=batch_size, drop_last=False)
      }
      if opt_str == 'adam':
        opt_class = Adam
      elif opt_str == 'sgd':
        opt_class = SGD
      
      optimizer = opt_class(model.parameters(), lr=lr)
      writer = SummaryWriter('./logs/cnn/{}'.format(number))
      a,b,c,d,e,f = train_CNN(model, dataloaders, optimizer, writer, n_epochs, ckpt_path, DEVICE)
      return a,b,c,d,e,f  
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#main function to execute code
if __name__=='__main__':
  #holdout validation phase
  x,y = cnn_main_val()
  n_hidden = 0
  optimizer = ''
  lr = 0.0
  print("validation for cnnClassifer")
  print("hidden_layer-32, optimizer - adam, learning rate-0.001 : train_loss{}  eval_loss{}".format(x[0],y[0]))
  print("hidden_layer-32, optimizer - sgd, learning rate-0.1 : train_loss{}  eval_loss{}".format(x[1],y[1]))
  print("hidden_layer-32, optimizer - sgd, learning rate-0.01 : train_loss{}  eval_loss{}".format(x[2],y[2]))
  print("hidden_layer-64, optimizer - adam, learning rate-0.001 : train_loss{}  eval_loss{}".format(x[3],y[3]))
  print("hidden_layer-64, optimizer - sgd, learning rate-0.1 : train_loss{}  eval_loss{}".format(x[4],y[4]))
  print("hidden_layer-64, optimizer - sgd, learning rate-0.01 : train_loss{}  eval_loss{}".format(x[5],y[5]))
  if (y.index(min(y))==0):
    n_hidden = 32
    optimizer = 'adam'
    lr = 0.001
  elif(y.index(min(y))==1):
    n_hidden = 32
    optimizer = 'sgd'
    lr = 0.1
  elif (y.index(min(y))==2):
    n_hidden = 32
    optimizer = 'sgd'
    lr = 0.01
  elif (y.index(min(y))==3):
    n_hidden = 64
    optimizer = 'adam'
    lr = 0.001
  elif(y.index(min(y))==4):
    n_hidden = 64
    optimizer='sgd'
    lr = 0.1
  else:
    n_hidden = 64
    optimizer='sgd'
    lr  =0.01
  print("parameters chosen are: {} {} {}".format(n_hidden,optimizer,lr))
  
  #testing phase
  x = 0
  testarr = []
  testarr2 = []
  testarr3 = []
  testarr4 = []
  testarr5 = []
  testarr6 = []
  while(x<5):
    a,b,c,d,e,f = cnn_main(n_hidden,optimizer,lr,x)
    testarr.append(a)
    testarr2.append(b)
    testarr3.append(c)
    testarr4.append(d)
    testarr5.append(e)
    testarr6.append(f)
    x = x+1

  print("Minimum train loss mean is {} and standard deviation is {}".format(cal_mean(testarr),cal_std(testarr)))
  print("Minimum evaluation loss mean is {} and standard deviation is {}".format(cal_mean(testarr4),cal_std(testarr4)))
  print("Maximum train top 1 accuracy Mean is {} and standard deviation is {}".format(cal_mean(testarr2),cal_std(testarr2)))
  print("Maximum evaluation top 1 accuracy Mean is {} and standard deviation is {}".format(cal_mean(testarr5),cal_std(testarr5)))
  print("Maximum train top 3 accuracy Mean is {} and standard deviation is {}".format(cal_mean(testarr3),cal_std(testarr3)))
  print("Maximum evaluation top 3 accuracy Mean is {} and standard deviation is {}".format(cal_mean(testarr6),cal_std(testarr6)))



