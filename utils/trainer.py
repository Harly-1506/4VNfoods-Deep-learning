import traceback
import tqdm
import os
import torch.nn as nn
import torchvision
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


Name_food = {
    0:"Banh mi",
    1:"Com tam",
    2:"Hu tieu",
    3:"Pho"
}

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def init_wandb(wb):
  #set up wandb for training
  if wb == True:
    try:
      print("------------SETTING UP WANDB--------------")
      import wandb
      wandb = wandb
      wandb = wandb.login()
      print("------Wandb Init-------")

      wandb.init(
          project = "MLP_FoodVN",
          name = "MLP_3hidden",
          config = {
              "batch_size" : 128,
              "learning_rate" : 0.0001,
              "epoch" : 50
          }
      )
      wandb.watch(model, criterion, log="all", log_freq=10)
      print()
      print("-----------------------TRAINING MODEL-----------------------")
      return wandb
    except:
        print("--------Can not import wandb-------")
  else: 
    wandb = None

def train(model,train_loader,optimizer, criterion, epoch, wandb = None, wb = False):

  model.train()
  train_loss = 0.0
  correct = 0
  total = 0
  train_acc = 0
  for i, (images, labels) in tqdm.tqdm(
        enumerate(train_loader), total = len(train_loader), leave = True, colour = "blue", desc = f"Epoch {epoch}",
        bar_format="{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    ):
      images = images.cuda(non_blocking = True)
      labels = labels.cuda(non_blocking = True)

      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      _, predicted = outputs.max(1)
      total += labels.size(0)
      correct += predicted.eq(labels).sum().item()

      metric = {
      " Loss" : train_loss / total,
      " Accuracy" :correct*100/ total,
      " epochs" : epoch,
      " Learning_rate" : get_lr(optimizer)
      }
      if wb == True and i <= len(train_loader):
        wandb.log(metric)
      
      # acc = correct*100 / total
      # train_acc += acc.item()
  acc = correct*100/total
  loss = train_loss/(i+1)

  
  print(" Loss: {:.4f}".format(loss), ", Accuracy: {:.2f}%".format(acc))


def val(model,valid_loader,optimizer, criterion, epoch, wandb = None, wb = False):

  model.eval()
  val_loss = 0.0
  correct = 0
  total = 0
  val_acc = 0
  with torch.no_grad():
    for i, (images, labels) in tqdm.tqdm(
          enumerate(valid_loader), total = len(valid_loader), leave = True, colour = "green", desc = "        ",
          bar_format="{desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
      ):
        images = images.cuda(non_blocking = True)
        labels = labels.cuda(non_blocking = True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        val_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        # acc = correct * 100/total
        # val_acc += acc.item()
    if wb == True:
      metric = {
          " Val_Loss" : val_loss/(i+1),
          " Val_Accuracy" :correct*100/total,
          # "Learning_rate" : self.learning_rate
      }
      wandb.log(metric)
    print(" Val_Loss: {:.4f}".format(val_loss/(i+1)), ", Val_Accuracy: {:.2f}%".format(correct*100/total))

  return correct*100/total
  

def test(model,test_loader,optimizer, criterion, epoch, wandb = None, wb = False):
  
  model.eval()
  test_loss = 0.0
  correct = 0
  total = 0
  test_acc = 0
  with torch.no_grad():
    for i, (images, labels) in tqdm.tqdm(
          enumerate(test_loader), total = len(test_loader), leave = True, colour = "blue", desc = " ",
          bar_format="{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
      ):
        images = images.cuda(non_blocking = True)
        labels = labels.cuda(non_blocking = True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # acc = correct*100 / total
        # test_acc += acc.item()
    if wb == True:
      wandb.log({"Test_accuracy": correct*100/total})

    print("--------TESTING--------")
    print(" Test_Loss: {:.4f}".format(test_loss/(i+1)), ", Test_Accuracy: {:.2f}%".format(correct*100/total))

from typing_extensions import TypeVarTuple

def save_weights(model, checkpoint_path):
  state_dict = model.state_dict()

  state = {
      "net" : state_dict
  }

  torch.save(state, checkpoint_path)

def fit(model,train_loader, valid_loader, test_loader, max_epochs = 50, max_plateau_count = 2, wb = False):

  
  epochs = 0
  plateau_count = 0
  val_acc_list = []
  best_acc_val = 0.0

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
  # optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.001, momentum = 0.9)
  # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
  scheduler = ReduceLROnPlateau(
    optimizer,
    patience=2,
    min_lr=0,
    # factor = torch.exp(torch.Tensor([-0.1])),
    verbose=True,
    factor = 0.1,
  )


  if wb == True:
    import wandb 
    wandb = wandb
    wandb.login()
    # wandb = init_wandb(wb)
    wandb.init(
        project = "classifi_FoodVN",
        name = "miniVGG_adam_l2_lr_0.001",
        config = {
            "batch_size" : 128,
            "learning_rate" : 0.0001,
            "epoch" : 50
        }
    )
    wandb.watch(model, criterion, log="all", log_freq=10)
  else:
    wandb = None
  checkpoint_dir = "/content/"

  checkpoint_path = os.path.join(checkpoint_dir, "{}_{}".format
                                      ("classification", "name_model"))

  try:
    while not plateau_count > max_plateau_count or epochs > max_epochs:
      epochs += 1
      train(model,train_loader,optimizer, criterion, epochs, wandb = wandb, wb = wb)
      val_acc = val(model,valid_loader,optimizer, criterion, epochs, wandb = wandb, wb = wb)

      val_acc_list.append(val_acc)

      #save weight
      if val_acc_list[-1]> best_acc_val:
        best_acc_val = val_acc_list[-1]
        save_weights(model, checkpoint_path)
      else:
        plateau_count += 1
        
      scheduler.step(100 - val_acc)

  except KeyboardInterrupt:
    traceback.print_exc()
    pass

  try: 
    state = torch.load(checkpoint_path)
    model.load_state_dict(state["net"])
    
    print("Accuracy on Private Test:")
    test(model,test_loader,optimizer, criterion, epochs, wandb = wandb, wb= wb)
  except:
    traceback.prtin_exc()
    pass
