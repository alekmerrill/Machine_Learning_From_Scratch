#!/usr/bin/env python
# coding: utf-8

# In[8]:


from Car import car
import numpy as np
import csv
import copy
from random import randint
# In[9]:


def read_track(filename):
    """
    Read track into a 2-D array
    @param filename - the text file being read in
    @return - track in 2-D array, (0,0) is bottom left corner
    """
    track = []
    states = []
    with open(filename, 'r') as fd:
        all_lines = fd.readlines()
        for line in all_lines:
            row = line.replace('\n', '')
            length = [char for char in row]
            track.append(length)
            state = length.copy()
            for k in range(len(state)):
                stk = state[k]
                if stk == '#':
                    state[k] = -2
                elif stk == '.' or stk == 'S':
                    state[k] = -1
                elif stk == 'F':
                    state[k] = 0
            states.append(state)
    #track = track[::-1]#reverse to make bottom left corner [0,0]
    fd.close()
    return (track[1:], states[1:])


# In[ ]:


def read_policy(track, policy, actions, truck, crashtype):
    """
    Have car follow a prescribed policy from value iteration
    @param track - the track the truck is on
    @param policy - the policy to be followed
    @param truck - the car
    @param crashtype -reset or continue
    """
    startx, starty = truck.getSpot()
    x1, y1 = startx,starty
    x, y = startx, starty
    reset = (crashtype == 'reset')
    iterations = 0
    x0, y0 = 0,0
    done, crash, fail = False, False, False
    cost = 0
    moves = 0
    while track[y][x] != 'F':
        if moves > 10000:
            fail = True
            break
        moves += 1
        x, y = truck.getSpot()
        x0, y0 = x, y
        truck2 = truck.copy()
        
        try:
            act_ind = policy[y][x].index(max(policy[y][x]))
            act = actions[act_ind]
        except Exception:
            print('Car beyond boundary')
            policy[y0][x0][act_ind] = -5
            iterations += 1
            if reset:
                truck2 = car(startx, starty)
                x,y = startx, starty
            else:
                truck2 = car(x, y)
            continue
        truck2.turn(act[0])
        truck2.accel(act[1])
        truck2.go()
        iterations += 1
        x1,y1 = truck2.getSpot()
        if y1 > len(track) or x1 > len(track[0]):
            policy[y0][x0][act_ind] = -5
            iterations += 1
            crash = True
            if reset:
                truck2 = car(startx, starty)
                x,y = startx, starty
            else:
                truck2 = car(x0, y0)
            continue
        if x < x1:
            change = +1
        else:
            change = -1
        while x != x1:
            loc = track[y][x]
            policy[y0][x0][act_ind] = -5
            if loc == '#':
                cost -=2
                iterations += 1
                if reset:
                    truck = car(startx, starty)
                    x,y = startx, starty
                    crash = True
                    break
                else:
                    truck = car(x-change, y)
                    x = x-change
                    break
            elif loc == 'F':
                iterations += 1
                cost += 0
                done = True
                break
            else:
                cost -= 1
            x += change
        if done or crash:
            done = False
            crash = False
            continue
        if y < y1:
            change = +1
        else:
            change = -1
        while y != y1:
            loc = track[y][x]
            if loc == '#':
                cost -= 2
                policy[y0][x0][act_ind] = -5
                iterations  += 1
                if reset:
                    truck = car(startx, starty)
                    policy[y0][x0][act_ind] = -5
                    x,y = startx, starty
                    crash = True
                    break
                else:
                    truck = car(x, y-change)
                    y = y-change
                    crash = True
                    break
            elif loc == 'F':
                cost += 0
                done = True
                break
            else:
                cost -= 1
            y += change
        if done or crash:
            done = False
            crash = False
            continue
        lastact = act_ind
        truck = truck2
        
    return iterations, cost, moves


# In[ ]:


def curr_val(track, truck):
    """
    Return value of current location
    """
    xy = truck.getSpot()
    spot = track[xy[1]][xy[0]]
    if spot == '#':
        return -2
    elif spot == 'F':
        return 0
    else:
        return -1


# In[ ]:


def value(track, truck, states, actions, discount):
    """
    Return (action, value) that results in highest value starting at current state
    @param track - the track driven on
    @param truck - the car driving
    @param states - input values for the track
    @param stateval - the value for the location and action
    @param actions - all possible actions that can be taken
    @param discount - the discount rate
    """
    x, y = truck.getSpot()
    max_val = -1
    best_act = actions[randint(0, 26)]
    vx, vy = truck.getV()
    for action_index in range(len(actions)):
        action = actions[action_index]
        if (vx == 0 and vy == 0) and action[1] == 'no':
            break
        x0 = x
        y0 = y
        val = curr_val(track, truck)
        truck2 = truck.copy()
        truck2.turn(action[0])
        truck2.accel(action[1])
        truck2.go()
        newx, newy = truck2.getSpot()
        if newx >= len(states[0]) or newx < 0:
            return -1, None
        if newy >= len(states) or newy < 0:
            return -1, None
        while x0 != newx:
            loc = track[y][x0]
            if loc == '#':
                return (-2, action)
            elif loc == 'F':
                return (0, action)
            if x0 < newx:
                x0 += 1
            if x0 > newx:
                x0 -= 1
        while y0 != newy:
            loc = track[y0][newx]
            if loc == '#':
                return (-2, action)
            elif loc == 'F':
                return (0, action)
            if y0 < newy:
                y0 += 1
            if y0 > newy:
                y0 -= 1
        next_val = states[y0][x0]
        val += discount * next_val
        if val > max_val:
            max_val = val
            best_act = action
            states[y][x] = max_val
    return (max_val, best_act)


# In[17]:

#Notes for self and development:
#turn = ['l', 'r', 'no'] #Turning options
#accels = ['x', 'y', '-x', '-y', '-xy', 'x-y', '-x-y', 'xy', 'no'] #acceleration options
#options = [(a, b) for a in turn for b in accels] #all possible decisions
#Rewards for options dependent on velocity and location of car
#probs = [1/27 for a in turn for b in accels] #Every move has equal chance for ease
#print(probs)


# In[ ]:


def valIter(track, actions, state_vals, startx, starty, iterations, learn, discount, delta):
    """
    Value iteration algorithm, takes in a track
    @param track - track being value iterrated
    @param actions - list of all possible actions that can be taken
    @param state_vals - instant values for each location on the track
    @param startx - starting positionx
    @param starty - starting positiony
    @param iterations - total iterations allowed
    @param learn - learning rate
    @param discount - discount rate
    @param delta - determines if change is close enough to 0 to break before iteration done
    """
    display = True
    states = [[0] * len(track[0])]*len(track)
    policy = [[[None]*len(actions)]*len(track[0])]*len(track)
    for it in range(iterations):
        start = True
        change = 0
        for i in range(len(track[0])):
            for j in range(len(track)):
                v = state_vals[j][i]
                if track[j][i] == 'F':
                    policy[j][i][26] = 100
                    continue
                truck = car(i, j)
                if track[j][i] == 'S':
                    start = True
                else:
                    start = False
                maxval=0
                for act_ind in range(len(actions)):
                    spot = policy[j][i]
                    action = actions[act_ind]
                    if display:
                        print("X,Y Coordinates:")
                        print(str(i) + ", " + str(j))
                        print(action)
                        print("Current Value:")
                        print(maxval)
                    truck2 = truck.copy()
                    x0, y0 = i,j
                    
                    truck2.turn(action[0])
                    truck2.accel(action[1])
                    truck2.go()
                    newx,newy = truck2.getSpot()
                    if newy >= len(track) or newx >= len(track[0]) or track[j][i] == '#':
                        continue
                    while x0 != newx:
                        loc = track[y0][x0]
                        if loc == '#':
                            crash = True
                            break
                        elif loc == 'F':
                            max_val = 100
                            done = True
                            break
                        elif loc == 'S' and not start:
                            crash = True
                            val = -1000
                            break
                        if x0 < newx:
                            x0 += 1
                        if x0 > newx:
                            x0 -= 1
                    while y0 != newy:
                        if crash or done:
                            crash = False
                            done = False
                            break
                        loc = track[y0][newx]
                        if loc == '#':
                            crash = True
                            break
                        elif loc == 'F':
                            done = True
                            break
                        elif loc == 'S' and not start:
                            crash = True
                            val = -1000
                            break
                        if y0 < newy:
                            y0 += 1
                        if y0 > newy:
                            y0 -= 1
                    val = curr_val(track, truck) +  discount*states[newy][newx]
                    if start and action[1]== 'no':
                        continue
                    spot[act_ind] = val
                    if display:
                        print("New value for state-action pair:")
                        print(val)
                        display = False
                    if val >= maxval:
                        states[j][i] = val
                        val = maxval
                    crash = False
                    done = False
        return policy, it
                
        
    
    
    
    


# In[1]:


def QLearn(track, actions, startx, starty, iterations, t_iter, discount, rate, crashtype, delta):
    """
    Q-Learning algorithm, takes in a track
    @param track - track being value iterrated
    @param actions - list of all possible actions that can be taken
    @param startx - starting positionx
    @param starty - starting positiony
    @param iterations - total iterations allowed
    @param t_iter - iterations on a single policy
    @param discount - discount rate/gamma
    @param rate - learning rate
    @param crashtype - reset/continue
    @param delta - determines if exploration decay is at its minimum
    """
    display = True
    display_opt = True
    reset = (crashtype == 'reset')
    exp = 1
    truck = car(startx, starty)
    Q = [[[0] * len(actions)] * len(track[0])] * len(track)
    #Every possible action for each state
    rewards = []
    t = 0 #current period
    max_reward = 0
    best_policy = []
    for it in range(iterations):
        policy = []
        truck2 = truck.copy()
        curr_reward = 0
        xt, yt = startx, starty
        crash = False
        done = False
        for t in range(t_iter):
            truck3 = truck2.copy()
            x0,  y0 = xt, yt
            loc = Q[yt][xt] #current location actions
            if track[yt][xt] == 'F':
                policy.append(('no','no'))
                Q[yt][xt][26] = 0
                break
            max_act = loc.index(max(loc))
            max_val = loc[max_act]
            
            chance = np.random.uniform(0,1)
            #Select action at current state
            
                     
            if chance < exp:
                action_index = randint(0, len(actions) - 1)
                action = actions[action_index]
            else:
                action_index = max_act
                action = actions[max_act]
            #while loc[action_index] < 0:
            #    action_index = randint(0, len(actions) - 2)
            #    action = actions[action_index] 
                
                
            if display:
                print("Original State-Action Pair:")
                print("(" + str(xt) + ", " + str(yt) + ")")
                print(action)
                print("Original Value: ")
                print(0)
            #print(action)
            #if loc[action_index] == -100:
            #    continue
            #Get value of this action, compare
            #If worse, use current best action, else use new action
            truck3.turn(action[0])
            truck3.accel(action[1])
            truck3.go()
            xy = truck3.getSpot() #next state
            if x0 < xy[0]:
                change = +1
            else:
                change = -1
            #determine if car crashes while driving
            while x0 != xy[0]:
                locs = track[y0][x0]
                if locs == '#':
                    Q[yt][xt][action_index] = -2
                    crash = True
                    policy.append(action)
                    if reset:
                        truck2 = truck.copy()
                        xt,yt = startx,starty
                    else:
                        truck2 = car(xt, yt)
                    break
                elif locs == 'F':
                    Q[yt][xt][action_index] = 0
                    policy.append(action)
                    done = True
                    break
                x0 = x0 + change
            if y0 < xy[1]:
                change = +1
            else:
                change = -1
            while y0 != xy[1]:
                if crash == True or done == True:
                    break
                locs = track[y0][xy[0]]
                if locs == '#':
                    Q[yt][xt][action_index] = -2
                    crash = True
                    policy.append(action)
                    if reset:
                        truck2 = truck.copy()
                        xt,yt = startx,starty
                    else:
                        policy.append(action)
                        truck2 = car(xt, yt)
                    break
                elif locs == 'F':
                    Q[yt][xt][action_index] = 0
                    policy.append(action)
                    done = True
                    break
                y0 = y0 + change
                    
            if crash == True:
                crash = False
                continue
            if done == True:
                break
            #if (xy[0] < 0) or (xy[1] < 1) or (xy[1] >= len(Q)) or (xy[0] >= len(Q[0])):
            #    Q[yt][xt][action_index] = -1
            #    truck2 = truck
            #    break
            #if track[xy[1]][xy[0]] == '#':
            #    Q[yt][xt][action_index] = -1
            #    truck2 = truck
            #    break
            
            spot_val = curr_val(track, truck) #Value of current location
            value = (1-rate)*loc[action_index] #Value update at Q Table
            try: #Done to deal with case where car goes off map
                value += rate*(spot_val + rate*max(Q[xy[1]][xy[0]]))
            except Exception:
                crash = True
                break
            if display:
                print("New Value for State-Action Pair:")
                print(value)
                display = False
            loc[action_index] = value
            truck2 = truck3
            curr_reward = value + curr_reward
            policy.append(action)
            xt = xy[0]
            yt = xy[1]
            if curr_val(track, truck2) >= 0:
                done = True
                break #Break loop if hit finish line
                
        #Update exploration probability        
        exp = max(delta, np.exp(-discount*it))
        if crash == True:
            crash = False
            continue
        if done == True:
            best_policy = policy
            break
        elif curr_reward >= max_reward:
            best_policy = policy
    return best_policy, it, len(best_policy)
        
    
    
    


# In[ ]:


def SARSA(track, actions, startx, starty, iterations, t_iter, discount, rate, delta, crashtype):
    """
    SARSA algorithm, takes in a track
    @param track - track being value iterrated
    @param actions - list of all possible actions that can be taken
    @param startx - starting positionx
    @param starty - starting positiony
    @param iterations - total iterations allowed
    @param t_iter - iterations on a single policy
    @param discount - discount rate/gamma
    @param rate - learning rate
    @param delta - minimum exploration decay
    @param crashtype - continue/reset
    """
    display = True
    display2 = True
    reset = (crashtype == 'reset')
    exp = 1
    truck = car(startx, starty)
    Q = [[[0] * len(actions)] * len(track[0])] * len(track)
    #Every possible action for each state
    rewards = []
    t = 0 #current period
    max_reward = 0
    best_policy = []
    for it in range(iterations):
        policy = []
        truck2 = truck.copy()
        curr_reward = 0
        xt, yt = startx, starty
        crash = False
        done = False
        chance = np.random.uniform(0,1)
        #Select first move, exclude move where nothing is done
        if chance < exp:
            action_index = randint(0, len(actions) - 1)
            action = actions[action_index]
            if display:
                print("Chance < Minimum Chance?")
                print(str(chance) + " < " + str(exp))
                print('Randomly generate action:')
                print(action)
        else:
            action_index = Q[starty][startx].index(max(Q[starty][startx]))
            action = actions[action_index]
            
            
        for t in range(t_iter):
            #print(action)
            truck3 = truck2.copy()
            x0,  y0 = xt, yt
            loc = Q[yt][xt] #current location actions
            if track[yt][xt] == 'F':
                done = True
                policy.append(('no','no'))
                Q[yt][xt][26] = 0
                break
            max_act = loc.index(max(loc))
            max_val = loc[max_act]
            
            #while loc[action_index] < 0:
            #    action_index = randint(0, len(actions) - 2)
            #    action = actions[action_index] 
                
                
                
            #print(action)
            #if loc[action_index] == -100:
            #    continue
            #Get value of this action, compare
            #If worse, use current best action, else use new action
            truck3.turn(action[0])
            truck3.accel(action[1])
            truck3.go()
            xy = truck3.getSpot() #next state
            
            #print(action)
            #print(xy)
            
            #determine if car crashes while driving
            while x0 != xy[0]:
                locs = track[y0][x0]
                if display:
                    print("Original Location:")
                    print("(" + str(x0) + ", " + str(y0) + ")")
                if locs == '#':
                    if display2:
                        print("Crashed into a wall at:")
                        print("(" + str(x0) + ", " + str(y0) + ")")
                    #print('skeet')
                    Q[yt][xt][action_index] = -2
                    curr_reward += rate * -2
                    crash = True
                    if reset:
                        if display2:
                            print("Resetting to starting line...")
                        truck2 = truck.copy()
                        xt,yt = startx,starty
                        if display2:
                            print('Back at start:')
                            print("(" + str(xt) + ", " + str(yt) + ")")
                            display2 = False
                    else:
                        truck2 = car(xt, yt)
                        if display2:
                            print("Placing car back on previous location...")
                            print("New location:")
                            print("(" + str(xt) + ", " + str(yt) + ")")
                            display2 = False
                    break
                elif locs == 'F':
                    Q[yt][xt][action_index] = 100
                    policy.append(action)
                    done = True
                    break
                if x0 < xy[0]:
                    x0 += 1
                if x0 > xy[0]:
                    x0 -= 1
            while y0 != xy[1]:
                if crash == True or done == True:
                    break
                locs = track[y0][xy[0]]
                if locs == '#':
                    #print('skat')
                    Q[yt][xt][action_index] = -2
                    curr_reward += rate * -2
                    crash = True
                    if reset:
                        truck2 = truck.copy()
                        xt,yt = startx,starty
                    else:
                        truck2 = car(xt, yt)
                    break
                elif locs == 'F':
                    Q[yt][xt][action_index] = 0
                    policy.append(action)
                    done = True
                    break
                if y0 < xy[1]:
                    y0 += 1
                if y0 > xy[1]:
                    y0 -= 1
                    
            if crash == True:
                crash = False
                continue
            if done == True:
                print('skedad')
                print(action)
                policy.append(action)
                break
            #if (xy[0] < 0) or (xy[1] < 1) or (xy[1] >= len(Q)) or (xy[0] >= len(Q[0])):
            #    Q[yt][xt][action_index] = -1
            #    truck2 = truck
            #    break
            #if track[xy[1]][xy[0]] == '#':
            #    Q[yt][xt][action_index] = -1
            #    truck2 = truck
            #    break
            loc2 = Q[xy[1]][xy[0]]
            max2_act = loc2.index(max(loc2))
            max2_val = loc2[max2_act]
            chance = np.random.uniform(0,1)
            if chance < exp:
                if display:
                    print("Chance < Minimum Chance?")
                    print(str(chance) + " < " + str(exp))
                    print('Randomly generate action:')
                    print(action)
                action2_index = randint(0, len(actions) - 1)
                action2 = actions[action2_index]
            else:
                action2_index = max2_act
                action2 = actions[action2_index]
            spot_val = curr_val(track, truck) #Value of current location
            value = (1-rate)*loc[action_index] #Value update at Q Table
            #loc2 = Q[xy[1]][xy[0]]
            try: #Done to deal with case where car goes off map
                value += rate*(spot_val + rate*loc2[action2_index])
            except Exception:
                value += rate * (spotval -2 * rate)
            if display:
                print("Original location-action value:")
                print(spot_val)
                print("Updated Value:")
                print(value)
            loc[action_index] = value
            truck2 = truck3
            curr_reward = value + curr_reward
            policy.append(action)
            xt = xy[0]
            yt = xy[1]
            if display:
                print("New location: ")
                print("(" + str(xt) + ", " + str(yt) + ")")
            action_index = action2_index
            action = action2
            if curr_val(track, truck2) >= 0:
                done = True
                break #Break loop if hit finish line
                
        #Update exploration probability        
        exp = max(delta, np.exp(-discount*it))
        if crash == True:
            crash = False
            continue
        if done == True:
            best_policy = policy
            break
        elif curr_reward >= max_reward:
            best_policy = policy
    return best_policy, it, len(best_policy)

