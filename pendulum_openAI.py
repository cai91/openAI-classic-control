# Author: Luis F. Camarillo-Guerrero
# Date: 9 April 2019
# Description: Evolutionary algorithm for the Pendulum-v0 environment from OpenAI gym

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
def pend(agent):

    R=0
    env = gym.make('Pendulum-v0')
    obs = env.reset()
    model.set_weights(rollParams(agent,[3,10,5,1]))

    for t in range(1000): #Max score attainable (t=1000)

        action = model.predict(np.array([obs]))[0]
        obs, reward, done, info = env.step(action)
        R+=reward

        if done:
            return (R),
            break

# Create neural network model
model = Sequential()
model.add(Dense(10, input_dim=3, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))

# Evolution settings

# Creator
creator.create("FitnessMax", base.Fitness, weights=(1.0,))     
creator.create("Individual", list, fitness=creator.FitnessMax) 

# Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float",random.uniform,-1,1)                    
toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attr_float, 101)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  
toolbox.register("evaluate", pend)
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
ind=[-0.6078847613323486, 0.26145280317725095, -2.231266393391227, -0.08877631140511771, 0.17111811068868857, 0.6312794678510525, 0.6829702986381786, -3.3840702700493623, -0.6190974259655218, 0.6141414386912453, -25.10400908754794, -0.01534319862942074, -1.7594375380131042, -0.2710144475059813, -1.5736454985910526, 0.008981032646585094, 0.5739288669265796, 0.5629734591378719, 0.8482898709916598, -0.21006209169875098, -10.26069587255971, -0.03354441457833454, -0.745419257672107, -0.17615511632960734, 1.2642531916538324, 0.24911385620310936, -0.889733234735799, -1.9873602848220029, 0.0522818270198973, -0.9210274290521838, -0.19168715165919065, -0.26666027036721973, 3.4102585235473297, 0.06092770866544153, -0.8066284271384108, 0.39345169369801775, 1.3484954749460452, -0.61213077912598, -0.5973044961837932, -0.44783126959676656, 0.4758038832278739, -8.39549468102551, 0.5890704935401831, 3.569590677623454, 1.1142055803337858, -1.2360391018257482, -1.1879138552948478, 0.09851044365276349, -0.1188602828177191, 0.09811434816504819, -0.561733260235709, 1.051725273048442, 8.36072247241914, 0.5271082743236418, 0.40632401636241855, 2.8892506672288154, -0.39706882707947205, 0.27888131519369247, 0.10708833603082883, -1.2828801119033357, -3.3057040905432236, -2.1617553801230036, -0.6787847808522505, -0.1888171744478102, 0.33862022364869826, -2.500070519748833, -0.7215351960358731, -4.813504708533788, -1.2796307279912715, 0.29492410887857423, -0.4434589180422438, 0.987512283366389, -0.17708216913500685, 0.6744864702848784, 1.09732205421175, 1.1439607308764144, 22.106064945100183, 1.4949190901593525, -0.7707040077744869, -2.2855008795487133, 0.7368780990612525, -0.7587943285864501, 0.3153336999189348, 0.20653679533757022, -7.440775581028868, 0.5362272866565088, -0.49800271875337604, -0.5809867763309329, -0.5877333198219641, 2.2741345605599794, 2.087071473223128, 0.10805829242164261, 0.3995430169723062, 0.09244048482966823, -1.8978081511204044, 0.5310822284308464, -38.3987136300906, 5.187330516525995, -1.4043033480525404, 0.16408329215300316, -0.8112048601163512]

for i in range(10):
    pend(ind)
"""