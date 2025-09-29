import torch
import torch.nn as nn
import numpy as np
# import dynamics_learning_encoder.mlmodel_encoder as mlmodel_encoder
import mlmodel_encoder
import pickle
import os
device = torch.device("cuda:1")
from torch.utils.tensorboard import SummaryWriter

"""
Loading the pickle data
"""
with open('quadrotor_dynamics_3000_2_5_increased_winds.pkl', 'rb') as f:
    data = pickle.load(f)

options = {}
options['dim_velocity'] = 2
options['dim_wind'] = 2
options['dim_encoder'] = 2
options['dim_control'] = 2
options['num_epochs'] = 100
options['learning_rate'] = 2e-5
options['batch_size'] = 128

f = mlmodel_encoder.f(options).to(device)
g = mlmodel_encoder.g(options).to(device)
mu = mlmodel_encoder.mu(options).to(device)

params = list(f.parameters()) + list(g.parameters()) + list(mu.parameters())
optimizer = torch.optim.Adam(params=params, lr=options['learning_rate'])
criteria = nn.MSELoss()

def train_test_split(data, test_size=0.2):
    indices = np.arange(len(data['velocities']))
    np.random.shuffle(indices)

    test_size = int(len(data['velocities']) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    train_data = {k:v[train_indices] for k,v in data.items() if k != 'metadata'}
    test_data = {k:v[test_indices] for k,v in data.items() if k != 'metadata'}
    
    return train_data, test_data

train_data, test_data = train_test_split(data)

train_set = mlmodel_encoder.VariableWindDataset(
    torch.tensor(train_data['velocities'][:, :2]).to(torch.float32), 
    torch.tensor(train_data['controls'][:, :2]).to(torch.float32), 
    torch.tensor(train_data['winds'][:, :2]).to(torch.float32), 
    torch.tensor(train_data['accelerations'][:, :2]).to(torch.float32)
)

test_set = mlmodel_encoder.VariableWindDataset(
    torch.tensor(test_data['velocities'][:, :2]).to(torch.float32), 
    torch.tensor(test_data['controls'][:, :2]).to(torch.float32), 
    torch.tensor(test_data['winds'][:, :2]).to(torch.float32), 
    torch.tensor(test_data['accelerations'][:, :2]).to(torch.float32)
)

"""
NOTE : need to see if I should be shuffling or not. 
"""
Trainloader = torch.utils.data.DataLoader(
    dataset=train_set, 
    batch_size=options['batch_size'],
    shuffle=True,
    num_workers=4,
)

ValidationLoader = torch.utils.data.DataLoader(
    dataset=test_set, 
    batch_size=options['batch_size'],
    shuffle=True,
    num_workers=4,
)

writer = SummaryWriter()
print(f"TensorBoard logs will be saved to: {writer.log_dir}")

training_loss = []
validation_loss = []

for epoch in range(options['num_epochs']):
    epoch_loss = 0.0
    epoch_loss_val = 0.0
    num_batches = 0
    num_batches_val = 0

    f.train()
    g.train()
    mu.train()

    for batch in Trainloader:
        optimizer.zero_grad()

        velocity = batch['velocity'].to(device)
        control = batch['control'].to(device)
        wind = batch['winds'].to(device)
        acceleration = batch['acceleration'].to(device)

        batch_size = velocity.shape[0]

        f_output = f(torch.concatenate([velocity, mu(wind)], dim=-1))        
        g_output = torch.matmul(g(velocity).reshape(batch_size, 2, 2), control.unsqueeze(-1)).squeeze(-1)

        model_output = f_output + g_output
        loss = criteria(acceleration, model_output)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
    
    training_loss.append(epoch_loss / num_batches)
    writer.add_scalar('Loss/train', training_loss[-1], epoch)

    f.eval()
    g.eval()
    mu.eval()

    with torch.no_grad():
        for batch in ValidationLoader:
            velocity = batch['velocity'].to(device)
            control = batch['control'].to(device)
            wind = batch['winds'].to(device)
            acceleration = batch['acceleration'].to(device)

            batch_size = velocity.shape[0]

            f_output = f(torch.concatenate([velocity, mu(wind)], dim=-1))        
            g_output = torch.matmul(g(velocity).reshape(batch_size, 2, 2), control.unsqueeze(-1)).squeeze(-1)

            model_output = f_output + g_output
            loss = criteria(acceleration, model_output)

            epoch_loss_val += loss.item()
            num_batches_val += 1
    
    validation_loss.append(epoch_loss_val / num_batches_val)
    writer.add_scalar('Loss/test', validation_loss[-1], epoch)
    print(f"Epoch {epoch} completed. Training Loss: {epoch_loss/num_batches}, Validation Loss: {epoch_loss_val/num_batches_val}")
    if (epoch % 10) == 0:
        print("wind actual : 1 ", wind[1,:], " Encoder output 1 : ", mu(wind[1,:]))
        print("wind actual : 2 ", wind[4,:], " Encoder output 1 : ", mu(wind[4,:]))
    if(epoch == 0):
        mlmodel_encoder.save_model(f, g, mu,'run_4_encoder_increased_wind' + str(epoch+1) + '.pth', options)
    if (epoch + 1) % 10 == 0:
        mlmodel_encoder.save_model(f, g, mu,'run_4_encoder_increased_wind' + str(epoch+1) + '.pth', options)
        print(f"Models saved at epoch {epoch+1} to")