

import torch
import random
import numpy as np

#Setting up manual seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

import torch.nn as nn
from src.dataset import train_dataloader, validation_dataloader
from src.model import PokemonCardModelV1
from src.utils import train_step, validation_step, accuracy_fn, print_time
from tqdm.auto import tqdm 




#Device agnostic code 
device = "cuda" if torch.cuda.is_available() else "cpu"

#input variables
input_shape = 3
hidden_units = 10
output_shape = 1

#instantiating model 0

model_0 = PokemonCardModelV1(input_shape =input_shape, hidden_units = hidden_units,output_shape= output_shape).to(device)

#Setting up loss function and optimizer

loss_fn = nn.BCEWithLogitsLoss()

#optimizer = torch.optim.SGD(model_0.parameters(), lr = .001)- with lr .01: highest val acc 93.75%  but kept randomly spiking down and up | with lr .001: kepting getting stuck in the 50-60s
optimizer = torch.optim.Adam(params = model_0.parameters(), lr = .001)


#Measuring time
from timeit import default_timer as timer
train_start_timer = timer()

### Training the model 0 (CNN)

epochs = 15

for epoch in tqdm(range(epochs)):
    print(f"--------------------------\n Epoch: {epoch}")
    train_step(model = model_0,
               data_loader=train_dataloader,
               loss_fn= loss_fn,
               optimizer= optimizer,
               accuracy_fn = accuracy_fn,
               device = device)

    validation_step(model = model_0,
              data_loader = validation_dataloader,
              loss_fn = loss_fn,
              accuracy_fn= accuracy_fn,
              device=device)
    print("--------------------------")

train_end_timer = timer()

total_train_time = print_time(start = train_start_timer, end= train_end_timer, device= device)

#Saving first model
# from pathlib import Path

# Model_Path = Path("models")

# Model_Path.mkdir(parents= True, exist_ok= True)

# Model_Name = "pokemon_card_model_v1.pth"

# Model_Save_Path = Model_Path / Model_Name

# print(f"Saving model to {Model_Save_Path}")

# torch.save(obj=model_0.state_dict(), f = Model_Save_Path)