from torchvision import transforms
import torch
import download_dataset, dataloaders, model_builder, engine, custom_loss, utils

config_data = utils.load_config()

URL = config_data['url']
FILENAME = config_data['filename']
DIR = config_data['dir_name']
BATCH_SIZE = config_data['batch_size']
RES_NET = config_data['res_net']
LEARNING_RATE = config_data['l_r']
MOMENTUM = config_data['momentum']
WEIGHT_DECAY = config_data['weight_decay']
EPOCHS = config_data['epochs']
MODEL_NAME = config_data['name']

download_dataset.download_dataset(URL,FILENAME)

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((448, 448)),
    transforms.ColorJitter(
        brightness=(1/1.5, 1.5),
        saturation=(1/1.5, 1.5))])

train_loader, test_loader = dataloaders.get_dataloaders(DIR, BATCH_SIZE, transforms)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model_builder.YOLO(RES_NET).to(device)

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr = LEARNING_RATE,
                            momentum = MOMENTUM,
                            weight_decay = WEIGHT_DECAY)

criterion = custom_loss.loss()

engine.train(EPOCHS, model, optimizer, criterion, train_loader, test_loader, MODEL_NAME, device)
