import numpy as np
from numpy import linalg as la
import pandas as pd
import datetime
import random
import time
import sys
import os


SIDE = int(sys.argv[1])
NUM_RUNS = int(sys.argv[2])

HOME = (np.arange(SIDE**2)+1)%(SIDE**2)

# enter as a vector. reshape and hold as a matrix
class puzzle_state:
    def __init__(self, config):
        conf1 = config
        self.config = conf1.reshape(SIDE,SIDE)
        self.parent = None
        self.d = 0

    def __str__(self):
        pass

    def copy(self):
        state_vec = self.config.copy()
        state_vec = state_vec.reshape(SIDE**2,)
        new = puzzle_state(state_vec)
        new.parent = self.parent
        return new

# find the neighbors of a puzzle_state
def moves(par_state: puzzle_state):                         
    a = par_state.config
    i = np.argwhere(a == 0)[0][0]
    j = np.argwhere(a == 0)[0][1]

    # center
    if (i!=0 and i!=SIDE-1) and (j!=0 and j!=SIDE-1):
        u = par_state.copy()
        d = par_state.copy()
        l = par_state.copy()
        r = par_state.copy()

        key = u.config[i-1,j]
        u.config[i-1,j] = u.config[i,j]
        u.config[i,j] = key

        key = d.config[i+1,j]
        d.config[i+1,j] = d.config[i,j]
        d.config[i,j] = key

        key = l.config[i,j-1]
        l.config[i,j-1] = l.config[i,j]
        l.config[i,j] = key

        key = r.config[i,j+1]
        r.config[i,j+1] = r.config[i,j]
        r.config[i,j] = key

        u.parent = par_state
        d.parent = par_state
        l.parent = par_state
        r.parent = par_state

        u.d = par_state.d + 1
        d.d = par_state.d + 1
        l.d = par_state.d + 1
        r.d = par_state.d + 1
        return [u,d,l,r]
    
    # corners
    elif i==0 and j ==0:
        d = par_state.copy()
        r = par_state.copy()
        key = d.config[i+1,j]
        d.config[i+1,j] = d.config[i,j]
        d.config[i,j] = key
        key = r.config[i,j+1]
        r.config[i,j+1] = r.config[i,j]
        r.config[i,j] = key

        r.parent = par_state
        d.parent = par_state

        r.d = par_state.d + 1
        d.d = par_state.d + 1
        return [r,d]
    elif i==0 and j==SIDE-1:
        d = par_state.copy()
        l = par_state.copy()
        key = d.config[i+1,j]
        d.config[i+1,j] = d.config[i,j]
        d.config[i,j] = key

        key = l.config[i,j-1]
        l.config[i,j-1] = l.config[i,j]
        l.config[i,j] = key

        l.parent = par_state
        d.parent = par_state

        l.d = par_state.d + 1
        d.d = par_state.d + 1
        return [l,d]
    elif i==SIDE-1 and j==0:
        u = par_state.copy()
        r = par_state.copy()
        key = u.config[i-1,j]
        u.config[i-1,j] = u.config[i,j]
        u.config[i,j] = key

        key = r.config[i,j+1]
        r.config[i,j+1] = r.config[i,j]
        r.config[i,j] = key

        u.parent = par_state
        r.parent = par_state

        u.d = par_state.d + 1
        r.d = par_state.d + 1
        return [u,r]
    elif i==SIDE-1 and j==SIDE-1:
        u = par_state.copy()
        l = par_state.copy()
        key = u.config[i-1,j]
        u.config[i-1,j] = u.config[i,j]
        u.config[i,j] = key

        key = l.config[i,j-1]
        l.config[i,j-1] = l.config[i,j]
        l.config[i,j] = key

        l.parent = par_state
        u.parent = par_state

        l.d = par_state.d + 1
        u.d = par_state.d + 1
        return [u,l]

    # non-corner edges
    elif i==0 and(j!=0 and j!=SIDE-1):
        d = par_state.copy()
        l = par_state.copy()
        r = par_state.copy()
        key = l.config[i,j-1]
        l.config[i,j-1] = l.config[i,j]
        l.config[i,j] = key
        key = r.config[i,j+1]
        r.config[i,j+1] = r.config[i,j]
        r.config[i,j] = key
        key = d.config[i+1,j]
        d.config[i+1,j] = d.config[i,j]
        d.config[i,j] = key

        l.parent = par_state
        r.parent = par_state
        d.parent = par_state

        l.d = par_state.d + 1
        r.d = par_state.d + 1
        d.d = par_state.d + 1
        return [l,r,d]
    elif i==SIDE-1 and(j!=0 and j!=SIDE-1):
        u = par_state.copy()
        l = par_state.copy()
        r = par_state.copy()
        key = r.config[i,j+1]
        r.config[i,j+1] = r.config[i,j]
        r.config[i,j] = key
        key = l.config[i,j-1]
        l.config[i,j-1] = l.config[i,j]
        l.config[i,j] = key
        key = u.config[i-1,j]
        u.config[i-1,j] = u.config[i,j]
        u.config[i,j] = key

        l.parent = par_state
        r.parent = par_state
        u.parent = par_state

        l.d = par_state.d + 1
        r.d = par_state.d + 1
        u.d = par_state.d + 1
        return [l,r,u]
    elif (i!=0 and i!=SIDE-1) and j==0:
        u = par_state.copy()
        d = par_state.copy()
        r = par_state.copy()
        key = u.config[i-1,j]
        u.config[i-1,j] = u.config[i,j]
        u.config[i,j] = key
        key = r.config[i,j+1]
        r.config[i,j+1] = r.config[i,j]
        r.config[i,j] = key
        key = d.config[i+1,j]
        d.config[i+1,j] = d.config[i,j]
        d.config[i,j] = key

        u.parent = par_state
        d.parent = par_state
        r.parent = par_state

        u.d = par_state.d + 1
        d.d = par_state.d + 1
        r.d = par_state.d + 1
        return [u,d,r]
    elif (i!=0 and i!=SIDE-1) and j==SIDE-1:
        u = par_state.copy()
        d = par_state.copy()
        l = par_state.copy()
        key = u.config[i-1,j]
        u.config[i-1,j] = u.config[i,j]
        u.config[i,j] = key
        key = l.config[i,j-1]
        l.config[i,j-1] = l.config[i,j]
        l.config[i,j] = key
        key = d.config[i+1,j]
        d.config[i+1,j] = d.config[i,j]
        d.config[i,j] = key

        u.parent = par_state
        d.parent = par_state
        l.parent = par_state

        u.d = par_state.d + 1
        d.d = par_state.d + 1
        l.d = par_state.d + 1
        return [u,d,l]

def random_state():
    here = puzzle_state(HOME)
    for i in range(100):
        here = random.choice(moves(here))
    here.parent = None
    here.d = 0
    return here

def is_found(found_list: list, to_check: puzzle_state):
    for s in found_list:
        if np.array_equal(s.config, to_check.config):
            return True
    
    return False


def manhattan_total(arr):
    sum = 0
    for i in range(SIDE):
        for j in range(SIDE):
            val = arr[i, j]
            if val != 0:
                sum = sum + np.abs( int((val-1)/SIDE) - i) + np.abs( ((val-1)%SIDE) - j)
    return sum

def horz_inv(arr):
    sum = 0
    flat = arr.copy()
    flat = arr.reshape(SIDE**2,)
    for i in range(SIDE**2):
        for j in range(i+1,SIDE**2):
            if flat[i] > flat[j] and (flat[j]!=0) and (flat[i]!=0):
                sum = sum + 1
    return sum

def inversion_dist(arr):
    return horz_inv(arr) + horz_inv(arr.transpose())

def heur(s: puzzle_state, w1: float, w2: float):
    flattened = s.config.copy()
    flattened = flattened.reshape(SIDE**2,)

    return w1*s.d + w2*(manhattan_total(s.config) + inversion_dist(s.config) )
    

def get_path(end: puzzle_state):
    bwds = [end]
    here = end.copy()

    while here.parent is not None:
        bwds = np.append(bwds, here.parent)
        here = here.parent
    n = len(bwds)
    return np.flip(bwds)

def find_sol_and_write_metrics(init_state: puzzle_state, path: str, upper_limit: int, w1: float, w2: float):
    qew = [init_state]
    min_heur = heur(init_state, w1=w1, w2=w2)
    found_states = []
    num_states_explored = 1
    sol_found = False
    flat_init = init_state.config.copy()
    flat_init = flat_init.reshape(SIDE**2,)

    print('searching...\t\tsearch algorithm: wA-star\t\tw1=',w1,'w2=',w2)
    start_time = datetime.datetime.now()
    while sol_found is False:        
        here = qew[0]
        qew = np.delete(qew,0)
        found_states = np.append(found_states, here)

        # put nbrs in qew
        nbrs = moves(here) 
        for n in nbrs:
            if not is_found(found_states, n):
                qew = np.append(qew, n)

        qew = sorted(qew, key=lambda a: heur(a, w1=w1, w2=w2))

        num_states_explored = num_states_explored+1
        if num_states_explored%100==0:
            print(num_states_explored, 'states expolored.')
        if num_states_explored >= upper_limit:
            print('experiment timeout.\n')
            end_time = datetime.datetime.now()
            
            metrics = {'experiment_date': start_time.strftime('%Y/%m/%d'),
               'experiment_start_time': start_time.strftime('%H:%M:%S'),
               'algorithm': 'wA-star',
               'w_1': w1,
               'w_2': w2,
               'size': SIDE,
               'initial_state': ''.join(np.array2string(flat_init,edgeitems=2)[1:-1].split()),
               'initial_0_norm': int(la.norm( HOME - flat_init, 0 )),
               'initial_taxi_norm': manhattan_total(init_state.config),
               'initial_inv_norm': inversion_dist(init_state.config),
               'num_states_explored': num_states_explored,
               'path_length': -1,
               'runtime': end_time-start_time
               }
            df = pd.DataFrame([metrics])
            df.to_csv(path, index=False, mode='a', header=False)
            return 1

        flattened = here.config.copy()
        flattened = flattened.reshape(SIDE**2,)

        if int(la.norm(HOME - flattened, 0)) == 0:
            end_time = datetime.datetime.now()
            sol_found = True
            end = here
        
    print('Solution found after', num_states_explored, 'states explored.')

    print('Solution length:', end.d,'\n')

    metrics = {'experiment_date': start_time.strftime('%Y/%m/%d'),
               'experiment_start_time': start_time.strftime('%H:%M:%S'),
               'w_1': w1,
               'w_2': w2,
               'size': SIDE,
               'initial_state': ''.join(np.array2string(flat_init,edgeitems=2)[1:-1].split()),
               'initial_0_norm': int(la.norm( HOME - flat_init, 0 )),
               'initial_taxi_norm': manhattan_total(init_state.config),
               'initial_inv_norm': inversion_dist(init_state.config),
               'num_states_explored': num_states_explored,
               'path_length': end.d,
               'runtime': end_time-start_time
               }
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False, mode='a', header=True)#False)


path = '/Users/calebhill/Documents/misc_coding/search/variants.csv'

find_sol_and_write_metrics(random_state(), path=path, upper_limit=2500, w1=0, w2=1)

# print('\n****************************************************************************')
# print('Starting', NUM_RUNS, 'runs with puzzle size', SIDE, '\tA-star weight of', A_STAR_FACTOR,'\tfocus factor of', FOCUS_FACTOR)
# for i in range(NUM_RUNS):
#     initial_state = random_state()
#     print('---------------- Beginning problem number:', i,'----------------\n')
#     for st in heuristics:
#         fresh_copy = initial_state.copy()
#         find_sol_w_focus(fresh_copy, path, search_type=st, upper_limit=2500, focus_factor=FOCUS_FACTOR)
# print('---------------- Experiment finished. ----------------\n\n')
