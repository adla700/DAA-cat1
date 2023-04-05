import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

def function():
    data = pd.read_csv(r"C:\Users\Adhula\Downloads\ML\Bank_Personal_Loan_Modelling.csv")
    data.drop(['ID'], axis=1, inplace=True)
    x = data.drop(['Personal Loan'], axis=1).values
    y = data['Personal Loan'].values
    x = torch.tensor(x, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    y = y.to(torch.float64)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.25)
    return x_train, x_test, y_train, y_test

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(12, 10, bias=False)
        self.linear2 = torch.nn.Linear(10, 20, bias=False)
        self.linear3 = torch.nn.Linear(20, 1, bias=False)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x.float())
        x = self.relu(x.float())
        x = self.linear2(x.float())
        x = self.relu(x.float())
        x = self.linear3(x.float())
        x = self.sigmoid(x.float())
        return x

model = NN()
loss_function = torch.nn.MSELoss()

class ParticleSwarmOptimizer:
    def __init__(self, model, w, c1, c2, num_of_particles, decay, inputs, labels):
        self.model = model
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.num_of_particles = num_of_particles
        self.inputs = inputs
        self.labels = labels
        self.initialize_position()
        self.initialize_velocity()
        self.pbest = self.positions
        self.gbest = np.inf
        self.decay = decay
        
    def initialize_position(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        r1 = np.random.rand(self.num_of_particles, num_params)
        self.positions = (10*r1)-0.5
        
    def initialize_velocity(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        r2 = np.random.rand(self.num_of_particles, num_params)
        self.velocity = r2 - 0.5
        
    def find_pbest(self):
        for i in range(len(self.pbest)):
            if self.fitness(self.pbest[i]) > self.fitness(self.positions[i]):
                self.pbest[i] = self.positions[i]
                
    def find_gbest(self):
        for position in self.positions:
            if self.fitness(position) < self.fitness(self.gbest):
                self.gbest = position
                
    def new_velocity(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        r1 = np.random.rand(self.num_of_particles, num_params)
        r2 = np.random.rand(self.num_of_particles, num_params)
        self.velocity = (self.w*self.velocity) + (self.c1*r1*(self.pbest-self.positions)) + (self.c2*r2*(self.gbest-self.positions))
    
    def new_position(self):
        self.positions += self.velocity
        
    def fitness(self, weights):
        outputs = self.model(self.inputs.float())
        loss = torch.nn.functional.binary_cross
