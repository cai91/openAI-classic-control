# Author: Luis F. Camarillo-Guerrero
# Date: 9 April 2019
# Description: Evolutionary algorithm for the MountainCarContinuous-v0 environment from OpenAI gym

# General libraries
import random
import numpy as np
import matplotlib.pyplot as plt

# Importing gym
import gym

# Neural network libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

#Genetic algorithms libraries
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import deap

# Function to roll parameters
def rollParams(uWs,top):
    '''This function takes in a list of unrolled weights (uWs) and a list with the number of neurons per layer in the following format:
    [input,first_hidden,second_hidden,output] and returns another list with the weights rolled ready to be input into a Keras model
    describing a two hidden layer neural network'''

    rWs=[]
    s=0
    
    for i in range(len(top)-1):
        tWs=[]
        for j in range(top[i]):
            tWs.append(uWs[s:s+top[i+1]])
            s=s+top[i+1]
            
        rWs.append(np.array(tWs))
        rWs.append(np.array(uWs[s:s+top[i+1]]))
        s=s+top[i+1]

    return rWs


# Fitness function
def mCar_c(agent):

    R=0
    env = gym.make('MountainCarContinuous-v0')
    obs = env.reset()
    model.set_weights(rollParams(agent,[2,10,5,1]))

    for t in range(1000): #Max score attainable (t=1000)

        action = model.predict(np.array([obs]))[0]
        obs, reward, done, info = env.step(action)
        R+=reward

        if done:
            return (R),
            break

# Create neural network model
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))

# Evolution settings

# Creator
creator.create("FitnessMax", base.Fitness, weights=(1.0,))     
creator.create("Individual", list, fitness=creator.FitnessMax) 

# Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float",random.uniform,-1,1)                    
toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attr_float, 91)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  
toolbox.register("evaluate", mCar_c)
toolbox.register("mate", tools.cxBlend,alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Launch evolutionary algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.2, ngen=50,  
                                   stats=stats, halloffame=hof, verbose=True)
    print('\nBest: ')
    print(hof)

main()

"""
# Comment out main(), add env.render() inside the loop of the fitness function and uncomment following code to see best agent in action.

# Best agent
ind=[4.81010520051896, 0.5299043643166147, 0.2998195017407944, 0.15948496124821482, -1.3579800371912674, -0.5493153897461533, 0.8094883626544327, -1.8204683039233067, -1.1288452737770411, -0.6171600054643518, -0.4397835424372782, -7.629580238264784, 0.6612135300617707, -1.251386773629764, -0.03926187100932055, 1.6193982807872231, -1.2851897800298817, 2.445156650349325, -0.10052087008303813, 3.165334835539768, 1.8306291985115446, -0.5382679417818419, -0.8797680804250766, -0.29725724885932825, 0.12223461666384756, 1.061181959242585, 0.2990640662609969, -0.552974253704578, -0.8128343836722244, -1.6107980809158255, -0.6217276681276993, -1.2372086085332006, -0.4011499051098096, -0.46192421714827075, -0.18665856990697993, 0.11150095096686438, 0.18571458400023977, -0.610799641984847, 0.03801676327495909, 0.5852496281681128, 0.36410454802335007, 1.8916342712863714, 0.5538539025946064, -0.6129152589219621, -0.79525594964709, 1.5473521733551099, 0.10575670994137831, 0.07157488670193696, -0.7807901200548454, 0.7653899457068913, 0.45116330693761475, 2.8253830917150293, -0.5545523155587938, 0.1859559183492791, -1.0848414456722324, -1.0003521872465624, -0.8427333585907236, -0.9991612251162041, -2.03415311507277, -0.4808982585164126, 0.057414736505936266, -0.898902906355133, -0.11317079548618157, -2.1440640055824307, 0.2879097499207464, 10.1914536739831, 0.9601135981579162, -0.549122218565489, -0.06274954836005284, -0.5674797417438127, -0.27483077997193206, -0.1155796263897188, 1.4174821548123018, -0.380305536671527, 0.006325529920494524, 0.5986582749064746, -0.5610672877605698, 0.8737133116848151, -0.3425423271189033, 0.404360620173924, -1.0397180309124012, -1.0524012551431214, 0.7192304680220563, -0.6510664675962172, -0.40529001546844473, 0.45092273622610446, -1.157820929703211, -0.6379273198328538, 0.1071656719705813, 0.23158769384726002, 8.978970557222437e-08]

for i in range(10):
    mCar_c(ind)
"""
