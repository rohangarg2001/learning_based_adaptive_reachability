import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import collections

Model = collections.namedtuple('Model', ['f', 'g', 'mu', 'options'])

class f(nn.Module):
    def __init__(self, options):
        super(f, self).__init__()
        self.options = options
        self.fc1 = nn.Linear(options['dim_velocity'] + options['dim_encoder'], 16) 
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, options['dim_velocity'])
        self.relu = nn.ReLU()
    
    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        
        return output

class g(nn.Module):
    def __init__(self, options):
        super(g, self).__init__()
        self.options = options
        self.fc1 = nn.Linear(options['dim_velocity'], 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, options['dim_velocity'] * options['dim_control'])
        self.relu = nn.ReLU()
    
    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        
        return output

class mu(nn.Module):
    def __init__(self, options):
        super(mu, self).__init__()
        self.options = options
        self.fc1 = nn.Linear(options['dim_wind'], 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, options['dim_encoder'])
        self.relu = nn.LeakyReLU(negative_slope=0.1)
    
    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)

        return output

class VariableWindDataset(Dataset):
    def __init__(self, velocity, control, wind, acceleration):
        self.velocity = velocity
        self.control = control
        self.wind = wind
        self.acceleration = acceleration
    
    def __len__(self):
        return self.velocity.shape[0]
    
    def __getitem__(self, index):
        velocity = self.velocity[index]        
        control = self.control[index]
        wind = self.wind[index]
        acceleration = self.acceleration[index]

        return {
            "velocity" : velocity, 
            "control" : control,
            "winds" : wind,
            "acceleration" : acceleration
        }

def save_model(f: f, g : g, mu: mu , model_name, options):
    if os.path.isdir('./models_encoders/') == False:
        os.makedirs('./models_encoders/')
    
    torch.save({
        'f_state_dict' : f.state_dict(),
        'g_state_dict' : g.state_dict(),
        'mu_state_dict' : mu.state_dict(),
        'options' : options
    }, './models_encoders/' + model_name + '.pth')

def load_model(modelname, model_path='./models/'):
    checkpoint = torch.load(model_path + '/' + modelname + '.pth')

    options = checkpoint['options']
    f_model = f(options)
    g_model = g(options)
    mu_model = mu(options)
    f_model.load_state_dict(checkpoint['f_state_dict'])
    g_model.load_state_dict(checkpoint['g_state_dict'])
    mu_model.load_state_dict(checkpoint['mu_state_dict'])

    return Model(f_model, g_model, mu_model, options)
