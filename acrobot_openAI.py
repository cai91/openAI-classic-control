# Author: Luis F. Camarillo-Guerrero
# Date: 9 April 2019
# Description: Evolutionary algorithm for the Acrobot-v1 environment from OpenAI gym

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
def acro(agent):

    R=0
    env = gym.make('Acrobot-v1')
    obs = env.reset()
    model.set_weights(rollParams(agent,[6,10,5,3]))

    for t in range(1000): #Max score attainable (t=1000)

        action = model.predict_classes(np.array([obs]))[0]
        obs, reward, done, info = env.step(action)
        R+=reward

        if done:
            return (R),
            break

# Create neural network model
model = Sequential()
model.add(Dense(10, input_dim=6, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

# Evolution settings

# Creator
creator.create("FitnessMax", base.Fitness, weights=(1.0,))     
creator.create("Individual", list, fitness=creator.FitnessMax) 

# Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float",random.uniform,-1,1)                    
toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attr_float, 143)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  
toolbox.register("evaluate", acro)
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
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.2, ngen=50,  
                                   stats=stats, halloffame=hof, verbose=True)
    print('\nBest: ')
    print(hof)

main()

"""
# Comment out main(), add env.render() inside the loop of the fitness function and uncomment following code to see best agent in action.

# Best agent
ind=[7.945959684794172, -3.3460930546211016, 0.11178139046139474, -0.5663723517302757, 0.11867227231114644, 0.27087942755371347, 0.8890517255671428, 0.9069876512054911, -0.09408591745326177, 0.3368792499019922, 0.1639568439621616, 0.018044852818249697, 12.537305111737442, 1.1485746413481908, -0.010728683163792621, -0.09775211224184906, -0.44382468582776885, 5.889381803907603, -0.05821380957731459, -1.1914536098174269, -0.8203119276706698, -0.11693852508699111, 0.2574232682508495, 0.22089795740047563, -0.5606366496994143, 1.0089184945090655, -2.4044481865072247, 1.4846303020188514, -1.0154593770179188, -0.36767995662656383, -2.06439682930544, -2.227955415375015, -0.8571638163826393, -1.8752928470298893, 3.3520557021239017, 1.1612460941596743, 0.19831003097273092, 0.005307135077572328, -1.004808187328205, 5.6217116024464735, -0.5871800277498884, 0.08690220004667178, -1.0077261022269084, -0.32259204119705454, 0.339569482530716, -0.661320065865302, -1.1402516480929294, 10.335829702947029, -1.2661558307784275, -1.835028820223073, -0.30556366388757306, 0.22328576855839713, 2.170352434938834, 0.3747411186870257, -1.50387149325481, 0.22341166085677003, -0.31583539168320357, -1.9075750716524653, -0.05259806163923952, -1.7843944520783508, 1.1060671301845988, 0.5565477455398076, 0.23263223011783443, 0.23279009254103716, -2.9286661339636035, 0.030399146852068085, -0.8340970253518549, 0.7329714757642771, -0.36878029816009344, -0.9529607445580517, 0.22448038974428125, -0.938717554777496, -0.41989799215112483, -0.7831104458575772, 1.0837284401131737, 1.3495714816083992, 0.9101970587028583, -0.5445916769856484, 2.072871445879197, 0.36807086414666296, -0.037609129235408434, 7.384960140680821, 2.7763349382920848, -1.3486866599514729, -2.78987619970951, 0.2560436440327819, -0.06084995747560076, -0.5580706025214919, -6.177815430626669, 0.4693927130066726, 0.22708904521970985, 1.5176233463901663, -2.6053869832611802, 3.4525983544627614, 0.9105861710310134, 0.2630323505113751, -0.42843598298877816, 1.2812925871037981, 0.3953675671290506, 0.9203817056898638, 2.5126259582970065, 0.8630734794425392, -0.13501117126496107, 0.9760080353635546, 0.6406598599159448, 0.837465185355822, -0.15121908930616326, -0.802323772108079, 0.5540208591606872, 0.7535521428260916, -1.1058376328745771, 0.6019412072342175, 4.522571355151643, 0.2278357431176357, -0.031474389381911896, 7.956477319449622, -0.3054510657119052, -0.4826144959644475, -0.21104418254611956, -1.304234598429352, 2.667357648148916, -0.860286828824216, 3.8302481569435542, -2.427778315936801, -0.013302967666273091, -0.578501440288189, -0.13637914917896699, -0.9926672072099552, -0.6017385256712457, -3.669733943164082, -0.10499971544858497, -1.8286594490909431, -2.1808137823510783, 0.13766215859635178, 0.37315515866336685, -1.6039742168169009, -1.7098161383271004, 0.24890284217397382, -1.4522013121576223, -0.16450016707679171, -0.8558922528234293, -0.3454042310326007, -0.15053127925041887]

for i in range(10):
    acro(ind)
"""
