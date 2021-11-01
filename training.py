import torch
from model import loss_fun
from utils import compute_accuracy
from tqdm.notebook import tqdm

def train_BT(train_loader, val_loader, model, optimizer, device, scaler, lmbda):  
  model.train()
  loop = tqdm(train_loader)
  total_loss = 0
  
  for batch_idx, (x1,x2 , _) in enumerate(loop):

    x1 = x1.to(device)
    x2 = x2.to(device)

    with torch.cuda.amp.autocast():
      z1, z2 = model(x1, x2)
      loss = loss_fun(z1, z2, lmbda)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    total_loss += loss

  #Evaluate for early stopping
  with torch.no_grad():
    model.eval()
    total_val_loss = 0
    for val_batch_idx, (x1_val , x2_val, _) in enumerate(val_loader):
      
      x1_val = x1_val.to(device)
      x2_val = x2_val.to(device)
      
      with torch.cuda.amp.autocast():
    
        z1_val, z2_val = model(x1_val, x2_val)
        loss_val = loss_fun(z1_val, z2_val, lmbda)
    
      total_val_loss += loss_val

  avg_loss = total_loss/(batch_idx+1)
  avg_val_loss = total_val_loss/(val_batch_idx+1)
    
  return avg_loss, avg_val_loss



def train_FT(train_loader, val_loader, ft_model, optimizer, criterion, device, scaler, lmbda ):  
  ft_model.train()
  loop = tqdm(train_loader)
  total_loss = 0
  
  for batch_idx, (x1 , _ , labels) in enumerate(loop):

    x1 = x1.to(device)
    labels = torch.tensor(labels,dtype=torch.long)
    labels = labels.to(device)
    
    with torch.cuda.amp.autocast():
      out = ft_model(x1)
      loss = criterion(out, labels)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    total_loss += loss

  avg_loss = total_loss/(batch_idx + 1)
  
  avg_val_loss = 0
  accuracy = 0
  
  #Evaluate and compute accuracy
  with torch.no_grad():
    ft_model.eval()
    accuracy = 0
    for val_batch_idx, (x_val , _, labels_val) in enumerate(val_loader):
      
      x_val = x_val.to(device)
      labels_val = torch.tensor(labels_val, dtype=torch.long).to(device)
      
      with torch.cuda.amp.autocast():
    
        out_val = ft_model(x_val)
        loss_val = criterion(out_val, labels_val)
        accuracy += compute_accuracy(out_val, labels_val)

      avg_val_loss += loss_val

  accuracy = accuracy/(val_batch_idx+1)
  avg_val_loss = avg_val_loss/(val_batch_idx+1)

  return avg_loss, avg_val_loss, accuracy