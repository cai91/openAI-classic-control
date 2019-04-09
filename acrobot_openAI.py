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
ind=[0.30053734287947936, 0.5388104609570843, 0.23395981769379817, 0.5018610993795476, -0.3243279864338898, 0.0009633537511318235, -0.5266916623658916, 0.275361736298686, -0.7718529311868535, 0.6452115311518245, -0.42136902442362584, -0.13151101325258072, 0.22589698495132776, -0.7433122464941692, -0.3479002336124667, 0.38243095947034816, 0.4317671240112252, -0.9840412126936304, -0.5695371305913225, -0.9831201003966286, -0.3947052045974075, 0.50054375105332, 2.6977579793392508, 0.13471614611365385, -1.792366902499221, 0.7651723229176454, -0.10764536268688374, -0.34290440263521527, -0.26433932669093163, 0.6920349377701132, -3.696448270476425, 0.7118807375972799, -0.06383223956694797, -0.22246804306501178, 0.3238301407999873, -0.09001192917683291, 0.02301326941198553, 2.086764028546183, 0.06931803171496276, 0.13335839056182483, 0.05049060648519219, -0.7541170594731794, -0.25145656122380466, 0.1442340969895644, 0.7995595627087776, 0.14939295419065354, 0.22753259087722189, 2.2017499270821386, 0.3584584971534265, 0.18212671096227695, -0.6378163856448256, -0.5335667506879065, 0.22784928986691289, 1.2117896541494886, 0.3697514282738814, 0.3526946134908588, 0.5863451832739414, -0.6587881258496645, -0.602526227001149, -0.03550388852780496, 0.3157059589435245, -0.28937212750009766, 0.10905846380237977, -1.084419126298869, 2.240680088475385, -0.2334977748648297, 0.11111988453262153, 0.8273461680754477, 0.17223732357039753, 0.9226470207414363, -0.7183552893270223, -0.607304180541817, 0.22156701570299464, -0.3761467354357594, -0.8119065842710624, 0.20783002090621777, -0.02626034347556318, -0.7340571281293529, -1.079857469870137, -0.3169572393182537, 0.16897040181984405, -0.005674025750446017, -0.27339911987030413, 0.8324195599668297, -0.3710972949557608, -0.7490347506704658, 0.6460677073438361, 0.0054712920218081, 0.6131279859998299, 0.6001916958500402, -0.1312759913029562, -0.5790600695967723, 0.35898265762073434, 0.45289572000944134, -0.5726422195948022, 1.8636846808036536, -0.18307594473345146, 0.5082868051613598, 0.4417055810534385, -0.31964831087637363, -0.7238220979166246, 0.41565560584514805, -0.09234203019120049, 0.3668792461895779, -0.1701452285543966, 0.6619594442630373, 0.8836546571256315, -1.6580325716266884, 0.009330696942161019, -0.0017996546269176683, 0.20256158126276447, 0.5382426343542623, -0.582374256399622, 0.4067210130939034, -0.29462865422529727, 1.1176096709760392, 0.12528322541197695, -0.5357263963620392, -0.38864886010344524, 0.2922465544226212, -0.5977778878280575, 0.25677590855337923, 0.10579164035823663, 0.3174201424381477, 0.12882556664962103, 0.45620855039627656, -0.10763429898661675, 0.31430329591333817, -0.17507433631124403, -0.8874010387880108, -1.0942024982121783, -0.09668369791663008, -0.4909404833502597, 0.7857993975710573, 0.03763906189077147, -0.5218281226120587, 1.1848521388834923, 0.10327824471083243, 0.905964891161855, -0.5973008296015253, -0.5981835828809582, -0.8218483138208921, 0.4342435388613708]

for i in range(10):
    acro(ind)
"""