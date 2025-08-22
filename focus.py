import numpy as np
from numpy import linalg as la
import pandas as pd
import datetime
import random
import time
import sys
import os

greedy_0 = 'greedy 0-norm'
a_star_0 = 'A-star 0-norm'

taxi_tot = 'taxicab total'
a_star_t = 'A-star,  taxi'

heuristics = [#greedy_0, 
              taxi_tot, 
              #a_star_0,
              a_star_t]

SIDE = int(sys.argv[1])
NUM_RUNS = int(sys.argv[2])
A_STAR_FACTOR = float(sys.argv[3])
FOCUS_FACTOR = float(sys.argv[4])


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

def heur(s: puzzle_state, search_type: str):
    flattened = s.config.copy()
    flattened = flattened.reshape(SIDE**2,)

    if search_type == greedy_0:
        return la.norm( HOME - flattened, 0 )
    elif search_type == a_star_0:
        return s.d + A_STAR_FACTOR*la.norm( HOME - flattened, 0 )
    elif search_type == taxi_tot:
        return manhattan_total(s.config)
    elif search_type == a_star_t:
        return s.d + A_STAR_FACTOR*manhattan_total(s.config)
    else:
        return 1

def get_path(end: puzzle_state):
    bwds = [end]
    here = end.copy()

    while here.parent is not None:
        bwds = np.append(bwds, here.parent)
        here = here.parent
    n = len(bwds)
    return np.flip(bwds)

def find_sol_w_focus(init_state: puzzle_state, path: str, search_type: str, upper_limit: int, focus_factor: float):
    qew = [init_state]
    focus = [init_state]
    min_heur = heur(init_state, search_type=search_type)
    found_states = []
    num_states_explored = 1
    sol_found = False
    flat_init = init_state.config.copy()
    flat_init = flat_init.reshape(SIDE**2,)
    if search_type==a_star_0 or search_type==a_star_t:
        weight = A_STAR_FACTOR
    else:
        weight = -1

    print('searching...\t\tsearch algorithm: focused', search_type)
    start_time = datetime.datetime.now()
    while sol_found is False:
        # pop focus
        if len(focus) > 0:
            here = focus[0]
            focus = np.delete(focus,0)
        else:
            here = qew[0]
            qew = np.delete(qew,0)
        found_states = np.append(found_states, here)

        # put nbrs in qew
        nbrs = moves(here) 
        for n in nbrs:
            if not is_found(found_states, n):
                qew = np.append(qew, n)
                if heur(n, search_type=search_type) <= FOCUS_FACTOR*min_heur:
                    focus = np.append(focus, n)

        qew = sorted(qew, key=lambda a: heur(a,search_type))
        min_heur = heur(qew[0], search_type=search_type)
        focus = sorted(focus, key=lambda a: heur(a,search_type))
        # sort focus

        num_states_explored = num_states_explored+1
        if num_states_explored%1000==0:
            print(num_states_explored, 'states expolored.')
        if num_states_explored == upper_limit:
            print('experiment timeout.\n')
            end_time = datetime.datetime.now()
            
            metrics = {'experiment_date': start_time.strftime('%Y/%m/%d'),
               'experiment_start_time': start_time.strftime('%H:%M:%S'),
               'algorihm': search_type,
               'a_star_factor': weight,
               'focus_factor': FOCUS_FACTOR,
               'size': SIDE,
               'initial_state': ''.join(np.array2string(flat_init,edgeitems=2)[1:-1].split()),
               'initial_0_norm': int(la.norm( HOME - flat_init, 0 )),
               'initial_taxi_norm': manhattan_total(init_state.config),
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

    p = get_path(end)
    print('Solution length:', len(p),'\n')

    metrics = {'experiment_date': start_time.strftime('%Y/%m/%d'),
               'experiment_start_time': start_time.strftime('%H:%M:%S'),
               'algorihm': search_type,
               'a_star_factor': weight,
               'focus_factor': FOCUS_FACTOR,
               'size': SIDE,
               'initial_state': ''.join(np.array2string(flat_init,edgeitems=2)[1:-1].split()),
               'initial_0_norm': int(la.norm( HOME - flat_init, 0 )),
               'initial_taxi_norm': manhattan_total(init_state.config),
               'num_states_explored': num_states_explored,
               'path_length': len(p),
               'runtime': end_time-start_time
               }
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False, mode='a', header=False)
    
    output = [end, num_states_explored, p, metrics]
    
    return output


path = '/Users/calebhill/Documents/misc_coding/search/focused_experimental_outputs.csv'

print('**************************************************')
print('Starting', NUM_RUNS, 'runs with puzzle size', SIDE, '\tA-star weight of', A_STAR_FACTOR,'\tfocus factor of', FOCUS_FACTOR)
for i in range(NUM_RUNS):
    initial_state = random_state()
    print('-------- Beginning problem number:', i,'--------\n')
    for st in heuristics:
        fresh_copy = initial_state.copy()
        find_sol_w_focus(fresh_copy, path, search_type=st, upper_limit=3000, focus_factor=FOCUS_FACTOR)
print('-------- Experiment finished. --------\n\n')

