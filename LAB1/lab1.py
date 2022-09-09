
import random
import monkdata as m
import numpy as np
import dtree as d
import matplotlib.pyplot as plt
from drawtree_qt5 import drawTree



# Assignment 1 Print entropy
monk1Entropy = d.entropy(m.monk1)
monk2Entropy = d.entropy(m.monk2)
monk3Entropy = d.entropy(m.monk3)
print('\nEntropy for monk data sets')
print(monk1Entropy,monk2Entropy,monk3Entropy)

# Assignment 3
dVec = [m.monk1, m.monk2, m.monk3]
print('\nGain for S on first level')
for i, data in enumerate(dVec):
    gain = []
    for atr in m.attributes:
        gain.append(round(d.averageGain(data,atr),6))

    print(gain)



# 5. 
print('\nGain for Sk on second level')
for value in m.attributes[4].values:
    gain = []
    Sk = d.select(m.monk1, m.attributes[4],value)

    for atr in m.attributes:
        gain.append(round(d.averageGain(Sk,atr),6))
    print(gain)
    
# Assignment 5
monk1Tree =d.buildTree(m.monk1, m.attributes)
monk2Tree =d.buildTree(m.monk2, m.attributes)
monk3Tree =d.buildTree(m.monk3, m.attributes)


print('Etrain = {} \t Etest = {}'.format(1-d.check(monk1Tree, m.monk1), 1-d.check(monk1Tree, m.monk1test)))
print('Etrain = {} \t Etest = {}'.format(1-d.check(monk2Tree, m.monk2), 1-d.check(monk2Tree, m.monk2test)))
print('Etrain = {} \t Etest = {}'.format(1-d.check(monk3Tree, m.monk3), 1-d.check(monk3Tree, m.monk3test)))

# Assignment 6

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def pruning(data,fraction):

    monkTrain, monkVal = partition(data, fraction)
    monkTree = d.buildTree(monkTrain,m.attributes)
    noPrune = d.check(monkTree,monkVal)
    monkCheck = noPrune
    improve = True
    while improve == True:
        monkPrune = d.allPruned(monkTree)
        best = 0
        for tree in monkPrune:
            partion = d.check(tree, monkVal)

            if partion > best:
                best = partion
                bestTree = tree

        if best < monkCheck: 
            improve = False
        else: 
            monkTree = bestTree
            monkCheck = best
    
    return monkCheck, noPrune

prunedTreeError = pruning(m.monk1, 0.6)[0]
print(prunedTreeError)

# Assignment 7 


fracVec =[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def error_stats(dataSet, iter):
    A = []
    APrune = []
    for frac in fracVec:
        Evec = []
        EvecPrune = []
        param = []
        pruneParam = []
        for i in range(iter):
            errors = pruning(dataSet, frac)
            Evec.append(1-errors[1])
            EvecPrune.append(1-errors[0])
        param.append(np.std(Evec))
        param.append(np.mean(Evec))
        pruneParam.append(np.std(EvecPrune))
        pruneParam.append(np.mean(EvecPrune))
        
        A = np.append(A,param)
        APrune = np.append(APrune,pruneParam)
    paramMat = np.reshape(A,(len(fracVec),len(param)))
    pruneParamMat = np.reshape(APrune,(len(fracVec),len(pruneParam)))
    return paramMat, pruneParamMat

iter = 1000
monk1Err, monk1ErrPrune = error_stats(m.monk1, iter)
plt.figure(1)
plt.plot(fracVec, monk1Err[:,1],'g*')
plt.plot(fracVec, monk1ErrPrune[:,1],'bo')
plt.ylabel('Mean errors')
plt.xlabel('Fractions')
plt.legend(['Without Pruning', 'With Pruning'])
plt.title('Error for MONK-1 with {} simulations.'.format(iter))

monk3Err, monk3ErrPrune = error_stats(m.monk3, iter)
plt.figure(2)
plt.plot(fracVec, monk3Err[:,1],'g*')
plt.plot(fracVec, monk3ErrPrune[:,1],'bo')
plt.ylabel('Mean errors')
plt.xlabel('Fractions')
plt.legend(['Without Pruning', 'With Pruning'])
plt.title('Error for MONK-3 with {} simulations.'.format(iter))

plt.show()

            






    


    
        




