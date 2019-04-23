import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
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

class MyMNIST(Dataset):
  def __init__(self, ds, flatten):
    self.flatten = flatten
    self.ds = ds
    
  def __len__(self):
    return len(self.ds)
  
  def __getitem__(self, index):
    image, label = self.ds[index]
    image = image.view(-1) if self.flatten else image
    return image, label


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

#autoencoder defined
class pretrainedAuto(nn.Module):
  def __init__(self):
    super().__init__()
    model33 = torch.load('pretrained_encoder.pt')['model']
    self.encoder =  model33
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1),
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=1),
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2),
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=8, out_channels=4,kernel_size=3, stride=1),
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=4, stride=2),
      nn.Sigmoid()
      )
    self.encoder = model33
    self.softmax = nn.Softmax(dim=1)
    #loss function set as mse loss
    self.loss_fn = nn.MSELoss(reduction='sum')
    #to transfer weights from pretrained encoder
    #reference: https://discuss.pytorch.org/t/copy-weights-only-from-a-networks-parameters/5841
    beta = 0.5 #The interpolation parameter    
    params1 = torch.load('pretrained_encoder.pt')['model'].named_parameters()
    params2 = self.named_parameters()
    dict_params2 = dict(params2)
    for name1, param1 in params1:
      if name1 in dict_params2:
        dict_params2[name1].data.copy_(beta*param1.data + (1-beta)*dict_params2[name1].data)
    self.load_state_dict(dict_params2)
  def forward(self,in_data):
    img_features = self.encoder(in_data)
    logits = self.decoder(img_features)
    return logits

  def loss(self, logits, labels):
    return self.loss_fn(logits, labels) / logits.size(0)

#train function for testing phase
#returns the minimum loss out of 20 epochs and the index of it(at which epoch) for testing
def train(model, loaders, optimizer, writer, n_epochs, ckpt_path, device='cpu'):
  def run_epoch(train_or_eval):
    epoch_loss = 0.
    epoch_acc = 0.
    for i, batch in enumerate(loaders[train_or_eval], 1):
      in_data, labels = batch
      in_data, labels = in_data.to(device), in_data.to(device)
      
      if train_or_eval == 'train':
        optimizer.zero_grad()
        
      logits = model(in_data)
      batch_loss = model.loss(logits, labels)
      
      epoch_loss += batch_loss.item()
      
      if train_or_eval == 'train':
        batch_loss.backward()
        optimizer.step()
        
        
    epoch_loss /= i
    
    losses[train_or_eval] = epoch_loss
    
    if writer is None:
      print('epoch %d %s loss %.4f acc %.4f' % (epoch, train_or_eval, epoch_loss, epoch_acc))
    elif train_or_eval == 'eval':
      writer.add_scalars('%s_loss' % model.__class__.__name__, # CnnClassifier or FcClassifier
                         tag_scalar_dict={'train': losses['train'], 
                                          'eval': losses['eval']}, 
                         global_step=epoch)
      
      if len(in_data.size()) == 2: # when it is flattened, reshape it
        in_data = in_data.view(-1, 1, 28, 28)
      if len(logits.size()) == 2: # when it is flattened, reshape it
        logits = logits.view(-1, 1, 28, 28)
          
      img_grid = make_grid(in_data.to('cpu'))
      out_img_grid = make_grid(logits.to('cpu'))
      writer.add_image('%s/eval_input' % model.__class__.__name__, img_grid, epoch)
      writer.add_image('%s/eval_output' % model.__class__.__name__, out_img_grid, epoch)
    return epoch_loss  
  # main statements
  losses = dict()
  #to store the loss for both training and testing
  arr1 = []
  arr2 = []
  for epoch in range(1, n_epochs+1):
    x = run_epoch('train')
    arr1.append(x)
    run_epoch('eval')
    arr2.append(x)
    # For instructional purpose, show how to save checkpoints
    if ckpt_path is not None:
      torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'losses': losses,
        #'accs': accs
      }, '%s/%d.pt' % (ckpt_path, epoch))
  #to return the 
  return min(arr2), arr2.index(min(arr2))

#train function for validation phase
#arr1, and arr2 to store MSE loss for each epoch on both training and testing
def train_val(model, loaders, optimizer, writer, n_epochs, ckpt_path, device='cpu'):
  arr1 = []
  arr2 = []
  def run_epoch(train_or_eval):
    epoch_loss = 0.
    epoch_acc = 0.
    for i, batch in enumerate(loaders[train_or_eval], 1):
      in_data, labels = batch
      in_data, labels = in_data.to(device), in_data.to(device)
      
      if train_or_eval == 'train':
        optimizer.zero_grad()
        
      logits = model(in_data)
      batch_loss = model.loss(logits, labels)
      
      epoch_loss += batch_loss.item()
      
      if train_or_eval == 'train':
        batch_loss.backward()
        optimizer.step()
        
        
    epoch_loss /= i
    
    losses[train_or_eval] = epoch_loss
    
    if writer is None:
      print('epoch %d %s loss %.4f acc %.4f' % (epoch, train_or_eval, epoch_loss, epoch_acc))
    elif train_or_eval == 'eval':
      writer.add_scalars('%s_loss' % model.__class__.__name__, # CnnClassifier or FcClassifier
                         tag_scalar_dict={'train': losses['train'], 
                                          'eval': losses['eval']}, 
                         global_step=epoch)
      
      
      if epoch % 10 == 0:
        if len(in_data.size()) == 2: # when it is flattened, reshape it
          in_data = in_data.view(-1, 1, 28, 28)
        if len(logits.size()) == 2: # when it is flattened, reshape it
          logits = logits.view(-1, 1, 28, 28)
          
        img_grid = make_grid(in_data.to('cpu'))
        out_img_grid = make_grid(logits.to('cpu'))
        writer.add_image('%s/eval_input' % model.__class__.__name__, img_grid, epoch)
        writer.add_image('%s/eval_output' % model.__class__.__name__, out_img_grid, epoch)
    return epoch_loss    
  
  # main statements
  losses = dict()
  
  for epoch in range(1, n_epochs+1):
    x = run_epoch('train')
    arr1.append(x)
    y = run_epoch('eval')
    arr2.append(y)
    # For instructional purpose, show how to save checkpoints
    if ckpt_path is not None:
      torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'losses': losses,
        #'accs': accs
      }, '%s/%d.pt' % (ckpt_path, epoch))
  arr1.sort()
  arr2.sort()
  return arr1[0],arr2[0]

#main for the testing phase
#returns the minimum MSE loss among 20 epochs and index(at which epoch) for the minimum value
def main(opt_test, lr_test):
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=lr_test)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--ckpt', type=str, default='./ckpt/autoencoder_test')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--optim', type=str, default=opt_test)
    args = parser.parse_args()
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
    model =  pretrainedAuto().to(DEVICE)
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
    writer = SummaryWriter('./logs/autoencoder_test/%s' % opt_str)
    min_val, min_index = train(model, dataloaders, optimizer, writer, n_epochs, ckpt_path, DEVICE)
    return min_val, min_index

#main function for validation phase
#executed for three times with 3 different parameter sets
#min_loss[] stores loss during training and min_loss2[] stores loss during testing
#retuns two arrays that store losses for training and testing
def main_val():
  val_param = [('adam',0.001),('sgd',0.1),('sgd',0.01)]
  min_loss = []
  min_loss2 = []
  x=0
  for a in val_param:    
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=a[1])
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--ckpt', type=str, default='./ckpt/autoencoder{}'.format(str(a[1])))
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--optim', type=str, default=a[0])
    args = parser.parse_args()
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
    model =  pretrainedAuto().to(DEVICE)
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
    writer = SummaryWriter('./logs/autoencoder/%s' % opt_str+str(lr))
    temp1, temp2 = train_val(model, dataloaders, optimizer, writer, n_epochs, ckpt_path, DEVICE)
    min_loss.append(temp1)
    min_loss2.append(temp2)
  return min_loss,min_loss2

#main function to execute code
#if the best metric decided form validation phase does not show the best reconstructed image
# comment out the validation phase from main function and choose the metrics manually by chaning optimizer and lr variables and execute program again
if __name__=='__main__':

#validation phase  
  x,y = main_val()
  optimizer = ''
  lr = 0.0
  print("validation for Convolutional Auto Encoder")
  print("optimizer - adam, learning rate-0.001 : train_loss{}  eval_loss{}".format(x[0],y[0]))
  print("optimizer - sgd, learning rate-0.1 : train_loss{}  eval_loss{}".format(x[1],y[1]))
  print("optimizer - sgd, learning rate-0.01 : train_loss{}  eval_loss{}".format(x[2],y[2]))
  if (y.index(min(y))==0):
    optimizer = 'adam'
    lr = 0.001
  elif(y.index(min(y))==1):
    optimizer = 'sgd'
    lr = 0.1
  elif (y.index(min(y))==2):
    optimizer = 'sgd'
    lr = 0.01
  print("parameters chosen are: {} {}".format(optimizer,lr))
  
#testing phase
  val_min, index_min = main(optimizer,lr)# if the best parameter chosen with progeam above does not have the best quality of reconstructed image, choose manually
  print("minimum value of mse loss is {} at {} th epoch".format(val_min,index_min+1))
