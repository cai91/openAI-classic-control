# Author: Luis F. Camarillo-Guerrero
# Date: 9 April 2019
# Description: Evolutionary algorithm for the MountainCar-v0 environment from OpenAI gym

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
def mCar(agent):

    R=0
    env = gym.make('MountainCar-v0')
    obs = env.reset()
    model.set_weights(rollParams(agent,[2,10,5,3]))

    for t in range(200): #Max score attainable (t=1000)

        action = model.predict_classes(np.array([obs]))[0]
        obs, reward, done, info = env.step(action)
        R+=reward

        if done:
            return (R),
            break

# Create neural network model
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

# Evolution settings

# Creator
creator.create("FitnessMax", base.Fitness, weights=(1.0,))     
creator.create("Individual", list, fitness=creator.FitnessMax) 

# Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float",random.uniform,-1,1)                    
toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attr_float, 103)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  
toolbox.register("evaluate", mCar)
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
ind=[-0.32733958437223626, 0.7489010427806485, -7.1366654943834495, 2.1810874603425816, -0.8675874991792499, 2.3764006813441023, -0.10128309748705633, -0.1983970225909275, 3.474410242687989, -4.154589767365746, -0.781286536267092, -7.126822080038949, -0.49512526576187454, -21.0134395368964, 2.6213220087376055, -1.8685223579355044, 0.4743315370116385, 0.5004412136959637, -0.14983638805173702, 3.5765753851479323, 19.02488867984934, -3.0445624203928685, -1.6892250538968931, 6.173489241738684, 6.20810728129182, -1.4217717435807151, -12.391063497220033, 14.925179735131, -0.5611800229271764, -0.7233399616465179, -2.378486106911947, 5.447422793174871, 0.3238071721763026, -26.366398403720257, -10.678953400940049, -7.723206773344568, 0.5235897659929657, 4.3698903911260025, -13.013768593021931, -0.36775563366153585, 0.6915987629545687, -0.847849845381663, 0.002704620389290752, -43.299936594389926, -10.996449368620578, 3.2185096699174416, -9.316978072384787, 31.82523276796253, 2.4454627029133644, 0.7158029243425649, -6.249638901331314, 2.9523896146329576, -8.46388371983436, -7.551082895197568, -1.0758948196029157, -6.210156704637344, -31.119225184585567, 1.3665254281435022, -17.901092427494707, -1.5183328079763327, 0.12786759254305186, 2.663030313278463, 1.3299773998624622, 1.5989055081998618, -18.77332186236395, 3.0481305229782274, 1.2102742791194974, -1.936803007787081, -4.569792827961726, 3.3437988566936245, 1.3520608450424563, 1.7117657433617826, 4.223452048574668, 0.4100091584510375, 0.051909603735688356, -4.424053964579185, 0.09303744726263169, 4.999536274572904, 1.1433730059306015, 3.1958116918478856, -2.83434483090733, 1.4713069609551492, 2.1523425099688764, -2.6771162043845678, -0.7449840612747206, 0.5390964737415472, -0.5902020969733512, -5.834216526023706, -0.8616466453855254, 2.587513242526514, 1.3655058507221596, 1.1399140006758994, -11.492971238932302, -0.8350080438233082, 0.11824759835626586, -12.816689337555623, 2.686854306381635, -2.4986684122926226, -0.5847911719773782, 0.6238492866357448, -1.5619064960247995, 1.13148613861703, 1.9013910787139292]

for i in range(10):
    mCar(ind)
"""