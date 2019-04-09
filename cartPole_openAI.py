# Author: Luis F. Camarillo-Guerrero
# Date: 9 April 2019
# Description: Evolutionary algorithm for the CartPole-v0 environment from OpenAI gym

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
def cartPole(agent):

    R=0
    env = gym.make('CartPole-v0')
    obs = env.reset()
    model.set_weights(rollParams(agent,[4,10,5,1]))

    for t in range(200): 

        action = model.predict_classes(np.array([obs]))[0][0]
        obs, reward, done, info = env.step(action)
        R+=reward

        if done:
            return (R),
            break

# Create neural network model
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Evolution settings

# Creator
creator.create("FitnessMax", base.Fitness, weights=(1.0,))     
creator.create("Individual", list, fitness=creator.FitnessMax) 

# Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float",random.uniform,-1,1)                    
toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attr_float, 111)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  
toolbox.register("evaluate", cartPole)
toolbox.register("mate", tools.cxBlend,alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Launch evolutionary algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.2, ngen=10, 
                                   stats=stats, halloffame=hof, verbose=True)
    print('\nBest: ')
    print(hof)

main()

"""
# Comment out main(), add env.render() inside the loop of the fitness function and uncomment following code to see best agent in action.

# Best agent
ind=[0.8786142609531805, -0.24499985914455918, 0.19830243346525878, 3.6135458449894977, -0.8756389008884015, -0.17625509764705652, 0.8238902515818699, -3.719170121907547, 0.9533418343262616, 1.3606989742046416, -0.10552819110606862, 0.07883893044152994, -0.13225081006020262, -0.8220883053882172, -1.1935508495598426, 0.1470175841665257, -0.11277111439210363, 1.1600658200085057, -0.8290904449138293, 0.37794766375458866, -0.9547008797729267, 1.0371657486039811, -2.17475178534988, 0.16401709691765404, -1.7635558440239587, -0.6324829440715244, -0.15417730908589997, -0.043030426038392036, -0.32573478851676185, 0.7038825332136986, -1.5318441728003125, -0.41884866442994606, -1.3524357666645215, 0.603326845672879, -0.03779834162305759, 0.7432539768279451, 1.1397187839825786, 1.327158088904039, 0.4998306704987332, -0.9104966036409607, 0.3307351133959217, 1.1052651459158027, 0.13110896569556038, 0.4091353922638391, -0.29229473505337566, -0.44761791653031524, 0.9206234610641171, 0.15283487657256578, 1.1768812206242758, -0.2117086642917545, -0.16998368121715757, 0.10990165849772918, -0.9162034681979427, -0.3578489891800434, 0.014799919774310579, -1.6628373102560525, -0.5916954478186089, -1.4734229345643473, -0.9762941207234848, -0.5268528685429597, 1.9362480535809967, -0.059642452067269466, 2.14489198215534, -0.901627145725414, -1.5442330143128042, 0.23779181664595178, 0.6088838140794923, -0.281336966549756, -0.11419085183349717, -0.37614406673447387, -0.21241909745889445, 0.3181345462145525, 0.46829460835323367, -0.3985292689276118, -0.758736232983492, -0.4003734659314951, -0.14871248132826265, 0.032694482647512374, -0.37913765462036075, 0.2671840830695918, -0.3752513355088073, -1.0889279936348917, -0.5800897618280139, 1.1201400900508431, -0.8216854784938163, -0.25736069605183853, 0.3704669268086817, 0.42367422374548486, 0.5448200834780769, -0.13254027403837923, 0.414567344742819, 0.15529735087176094, 0.5460743730241069, -0.7959490785624537, -1.1729591807542195, 0.9924160371034438, -2.4801505279912885, -0.08216604767737523, 1.2036801936608237, -0.04953529297306664, -0.1576094365745388, 0.8559486824645739, -0.834284341650256, 1.707193321358777, 0.5392708474345845, -0.7808348619320679, -0.026440866545192947, 0.043084343629598065, 1.2209150563395077, 0.46783104436520323, -0.8629512011905451]

for i in range(10):
    cartPole(ind)
"""