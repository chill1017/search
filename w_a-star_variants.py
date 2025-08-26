import numpy as np
import puzzle_utilities as util
import animate_solution

from numpy import linalg as la
import pandas as pd
import datetime
import sys

SIDE = int(sys.argv[1])
NUM_RUNS = int(sys.argv[2])

HOME = (np.arange(SIDE**2)+1)%(SIDE**2)



def is_found(found_list: list, to_check: util.puzzle_state):
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

def heur(s: util.puzzle_state, w1: float, w2: float):
    flattened = s.config.copy()
    flattened = flattened.reshape(SIDE**2,)

    return w1*s.d + w2*(manhattan_total(s.config) + inversion_dist(s.config) )
    

def get_path(end: util.puzzle_state):
    bwds = [end]
    here = end.copy()

    while here.parent is not None:
        bwds = np.append(bwds, here.parent)
        here = here.parent
    n = len(bwds)
    return np.flip(bwds)

def find_sol_and_write_metrics(init_state: util.puzzle_state, path: str, upper_limit: int, w1: float, w2: float):
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
        nbrs = util.moves(here) 
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
    df.to_csv(path, index=False, mode='a', header=False)

    return end



metrics_path = '/Users/calebhill/Documents/misc_coding/search/variants.csv'

find_sol_and_write_metrics(util.random_state(), path=metrics_path, upper_limit=2500, w1=0, w2=1)

# print('\n****************************************************************************')
# print('Starting', NUM_RUNS, 'runs with puzzle size', SIDE, '\tA-star weight of', A_STAR_FACTOR,'\tfocus factor of', FOCUS_FACTOR)
# for i in range(NUM_RUNS):
#     initial_state = random_state()
#     print('---------------- Beginning problem number:', i,'----------------\n')
#     for st in heuristics:
#         fresh_copy = initial_state.copy()
#         find_sol_w_focus(fresh_copy, path, search_type=st, upper_limit=2500, focus_factor=FOCUS_FACTOR)
# print('---------------- Experiment finished. ----------------\n\n')
