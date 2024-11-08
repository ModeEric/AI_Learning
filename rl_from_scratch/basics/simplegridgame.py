import random
gridsize = (10,10)

states = [[i,j for j in range(gridsize[1]) ]for i in range(gridsize[0])]
actions = [[0,1],[1,0],[-1,0],[0,-1]]
gamma = .9
alpha = .2
epsilon = 1

startingpos = (gridsize[0]//2,gridsize[0]//2)
pos = startingpos
endingpos = gridsize

def isinvalid(move,pos):
    pass
Q = [[[0 for i in range(len(actions))] for j in range(gridsize[1])] for k in range(gridsize[0])]
while(pos != endingpos):
    if random.random() > epsilon:
        move = random.choice(actions)
    else:
        move = #argmax action Q[pos[0][pos[1]][a]
    if isinvalid(move,pos):
        reward = -1
    else:
        reward = ((endingpos[0]-pos[0])**2 + (endingpos[1]-pos[1])**2 - (endingpos[0]-startingpos[0])**2 + (endingpos[1]-startingpos[1])**2)  / (endingpos[0]**2 + endingpos[1]**2)
    Q[pos[0]][pos[1]][move] = Q[pos[0]][pos[1]][move]+alpha*(reward+gamma*...) #argmax new action - Q[s,a]
    pos = (pos[0]+move[0],pos[1]+move[1])
print(Q)