import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
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
        self.linear1 = torch.nn.Linear(12, 10)
        self.linear2 = torch.nn.Linear(10, 20)
        self.linear3 = torch.nn.Linear(20, 1)
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

class GeneticOptimizer:
    def __init__(self, model, population_size, mutation_rate, decay_rate, inputs, labels):
        self.model = model
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.decay_rate = decay_rate
        self.population = self.init_population()
        self.inputs = inputs
        self.labels = labels

    def init_population(self):
        population = []
        for i in range(self.population_size):
            weights = []
            for weight in self.model.parameters():
                weights.append(weight.data.numpy())
            population.append(weights)
        return population

    def selection(self, fitness_scores):
        cumulative_scores = np.cumsum(fitness_scores)
        total_score = np.sum(fitness_scores)
        rand = np.random.uniform(0, total_score)
        selected_index = np.searchsorted(cumulative_scores, rand)
        return selected_index

    def crossover(self, male, female):
        random_crossover = np.random.randint(1, len(male))
        child1 = male[:random_crossover] + female[random_crossover:]
        child2 = female[:random_crossover] + male[random_crossover:]
        return child1, child2
    
    def decay_mutation_rate(self):
        self.mutation_rate -= (self.decay_rate * self.mutation_rate)

    def mutate(self, child):
        for i in range(len(child)):
            if np.random.uniform(0, 1) < self.mutation_rate:
                child[i] += np.random.normal(0, 0.1, child[i].shape)
        return child

    def generate_offspring(self, fitness_scores):
        new_population = []
        for _ in range(self.population_size):
            parent1_index = self.selection(fitness_scores)
            parent2_index = self.selection(fitness_scores)
            parent1 = self.population[parent1_index]
            parent2 = self.population[parent2_index]
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self
