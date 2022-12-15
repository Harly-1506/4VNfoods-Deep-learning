import traceback
import tqdm
import os
import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F

Name_food = {
    0:"Banh mi",
    1:"Com tam",
    2:"Hu tieu",
    3:"Pho"
}

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=5):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

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
  iou = 0.0
  total = 0
  train_acc = 0
  for i, (images, labels) in tqdm.tqdm(
        enumerate(train_loader), total = len(train_loader), leave = True, colour = "blue", desc = f"Epoch {epoch}",
        bar_format="{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    ):
      images = images.cuda(non_blocking = True)
      labels = labels.cuda(non_blocking = True)
      # labels = labels.ToTensor()

      optimizer.zero_grad()
      
      outputs = model(images)
      # print("out" , total)
      # print("type", mask.type())
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      correct += pixel_accuracy(outputs, labels)
      iou += mIoU(outputs, labels)
      metric = {
      " Loss" : train_loss / 32,
      " Accuracy" :correct*100/ 32,
      " epochs" : epoch,
      " Learning_rate" : get_lr(optimizer),
      "Iou_train":iou / len(train_loader)
      }
      if wb == True and i <= len(train_loader):
        wandb.log(metric)
      
      # acc = correct*100 / total
      # train_acc += acc.item()
  acc = correct/len(train_loader)
  iou_score = iou / len(train_loader)
  loss = train_loss/(i+1)

  
  print(" Loss: {:.4f}".format(loss), ", Accuracy: {:.2f}%".format(acc), ", IouSroce: {:.2f}%".format(iou_score))


def val(model,valid_loader,optimizer, criterion, epoch, wandb = None, wb = False):

  model.eval()
  val_loss = 0.0
  correct = 0
  total = 0
  val_acc = 0
  iou = 0
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
        correct += pixel_accuracy(outputs, labels)
        iou += mIoU(outputs, labels)
        # acc = correct * 100/total
        # val_acc += acc.item()
    if wb == True:
      metric = {
          " Val_Loss" : val_loss/(i+1),
          " Val_Accuracy" :correct*100/len(valid_loader),
          # "Learning_rate" : self.learning_rate
          "Iou_val": iou / len(valid_loader)
      }
      wandb.log(metric)
    iou_score = iou / len(valid_loader)
    print(" Val_Loss: {:.4f}".format(val_loss/(i+1)), ", Val_Accuracy: {:.2f}%".format(correct/len(valid_loader)), " IouScore: {:.2f}%".format(iou_score))

  return correct/len(valid_loader)
  

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
        correct += pixel_accuracy(outputs, labels)

        # acc = correct*100 / total
        # test_acc += acc.item()
    if wb == True:
      wandb.log({"Test_accuracy": correct*100/len(test_loader)})

    print("--------TESTING--------")
    print(" Test_Loss: {:.4f}".format(test_loss/(i+1)), ", Test_Accuracy: {:.2f}%".format(correct/len(test_loader)))

from typing_extensions import TypeVarTuple

def save_weights(model, checkpoint_path):
  state_dict = model.state_dict()

  state = {
      "net" : state_dict
  }

  torch.save(state, checkpoint_path)

def fit(model,train_loader, valid_loader, test_loader, max_epochs = 50, max_plateau_count = 2, wb = False):

  torch.cuda.empty_cache()
  epochs = 0
  plateau_count = 0
  val_acc_list = []
  best_acc_val = 0.0

  criterion = DiceLoss(mode = "multiclass")
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

  if wb == True:
    import wandb 
    wandb = wandb
    wandb.login()
    # wandb = init_wandb(wb)
    wandb.init(
        project = "MLP_FoodVN",
        name = "MLP_1hidden",
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
