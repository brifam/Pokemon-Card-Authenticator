import torch


def accuracy_fn(y_true,y_pred):

    """Calculates Model's Accuracy"""
    amount_correct = torch.eq(y_true,y_pred).sum().item()

    acc = (amount_correct/len(y_pred)) * 100

    return acc

def train_step(model: torch.nn, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn : torch.nn.Module, 
               optimizer : torch.optim.Optimizer, 
               accuracy_fn,
               device : torch.device = device):

    """Trains the model on the dataloader"""
    train_loss, train_accuracy = 0,0
    model.train()

    for (X,y) in data_loader:

        X,y = X.to(device), y.to(device)
        
        y_logits = model(X)
        y_pred = torch.round(torch.sigmoid(y_logits))

        loss = loss_fn(y_logits,y)
        train_loss += loss.item()

        accuracy = accuracy_fn(y,y_pred)
        train_accuracy +=accuracy


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


    train_loss /= len(data_loader)
    train_accuracy /= len(data_loader)

    print(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")


def validation_step(model: torch.nn, 
                    data_loader: torch.utils.data.DataLoader, 
                    loss_fn : torch.nn.Module, 
                    accuracy_fn, 
                    device : torch.device = device):
    
    val_loss, val_accuracy = 0,0
    model.eval()
    with torch.inference_mode():
        for (X,y) in data_loader:
            X,y = X.to(device), y.to(device)
            y_val_logits = model(X)
            y_val_pred = torch.round(torch.sigmoid(y_val_logits))

            loss = loss_fn(y_val_logits,y)
            val_loss+= loss.item()

            accuracy = accuracy_fn(y,y_val_pred)
            val_accuracy += accuracy


    val_loss /= len(data_loader)
    val_accuracy /= len(data_loader)

    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")



        


    





        
