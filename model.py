import torch, time
from torchvision import models 
from tqdm import tqdm


def train_model(model, criterion, optimizer, dataloaders, dataset_sizes,  num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = {"loss":[],"acc":[]}
    train = {"loss":[],"acc":[]}
    start_time = time.time()
    best_acc= 0.0
    for epoch in range(num_epochs):
        print("epoch{}/{}".format(epoch, num_epochs - 1))
        print("-" * 10) 
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0.0
            for inputs,labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs,1)
                    loss = criterion(outputs,labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == "train":
              train["loss"].append(epoch_loss)
              train["acc"].append(epoch_acc.item())
            else:
              val["loss"].append(epoch_loss)
              val["acc"].append(epoch_acc.item())

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss,epoch_acc))
            if phase == "val" and epoch_acc > best_acc:
                torch.save(model.state_dict(),"./models/mobilenet-v3-small-best.pth")
                best_acc = epoch_acc

    time_elapsed = time.time() - start_time
    print("training completed in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed%60))
    print("best val accuracy: {:4f}".format(best_acc))
    return model

def get_model():
    return models.mobilenet_v3_small(pretrained=True)
