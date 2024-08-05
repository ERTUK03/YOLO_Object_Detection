import torch
def train_step(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               criterion: torch.nn.Module,
               device: torch.device):

    running_loss = 0.
    last_loss = 0.

    model.train()

    for i, (images, annotations) in enumerate(train_dataloader):
        images = images.to(device)
        annotations = torch.tensor(annotations).to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, annotations)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss /10
            print(f'batch {i+1} loss: {last_loss}')
            running_loss = 0.

    return last_loss

def test_step(model: torch.nn.Module,
              test_dataloader: torch.utils.data.DataLoader,
              criterion: torch.nn.Module,
              device: torch.device):

    running_vloss = 0.
    model.eval()

    with torch.no_grad():
        for i, (images, annotations) in enumerate(test_dataloader):
            images = images.to(device)
            annotations = torch.tensor(annotations).to(device)
            outputs = model(images)
            vloss = criterion(outputs, annotations)
            running_vloss += vloss.item()
    avg_loss = running_vloss / (i + 1)
    return avg_loss

def custom_lr_scheduler(epoch):
    if epoch < 10:
        return 0.001 + (0.01 - 0.001) * (epoch / 10)
    elif epoch < 85:
        return 0.01
    elif epoch < 115:
        return 0.001
    else:
        return 0.0001

def train(epochs: int,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          model_path: str,
          device: torch.device):

    best_vloss = 1000000

    for epoch in range(1,epochs+1):
        lr = custom_lr_scheduler(epoch)
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr
        print(f'EPOCH {epoch} l_r {lr}')
        avg_loss = train_step(model, train_dataloader, optimizer, criterion, device)
        avg_vloss = test_step(model, test_dataloader, criterion, device)
        print(f'LOSS train {avg_loss} test {avg_vloss}')
        if avg_vloss<best_vloss:
            best_vloss = avg_vloss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_vloss
                }, f'{model_path}_{epoch}.pth')
