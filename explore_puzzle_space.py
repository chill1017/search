# idea: BFS explore the whole space and recreate the histograms for # of states with a given norm

# to get some benchmarks: after getting all states, choose some states outside a certain (taxi)^2 + (inv)^2 ellipse as benchmarks
# eccentricity of the ellipse could be mean/sd of norm distribution or something

import numpy as np
import puzzle_utilities as util

from numpy import linalg as la
import pandas as pd
import datetime
import sys

SIDE = int(sys.argv[1])
HOME = (np.arange(SIDE**2)+1)%(SIDE**2)

solved_state = util.puzzle_state(HOME)

def explore_and_record():
    frontier = [solved_state]
    discovered = set()
    discovered.add(solved_state)
    num_states_explored = 1
    state_data = []

    print('starting exploration...')
    while len(frontier) != 0:
        here = frontier[0]
        frontier = np.delete(frontier,0)

        flattened_here = here.config.reshape(SIDE**2,)

        nbrs = util.moves(here)
        for n in nbrs:
            if n not in discovered:
                discovered.add(n)
                frontier = np.append(frontier,n)

        state_data.append([
            num_states_explored, 
            la.norm( HOME - flattened_here ,0), 
            la.norm( HOME - flattened_here ,1), 
            round(la.norm( HOME - flattened_here ,2),3), 
            util.manhattan_total(here.config), 
            util.inversion_dist(here.config),
            ''.join(np.array2string(flattened_here,edgeitems=2)[1:-1].split())
            ])

        num_states_explored = num_states_explored + 1
        if num_states_explored%10000 == 0:
            print(num_states_explored,'states explored.')

    df = pd.DataFrame(columns=['num_explored', 'zero_norm', 'one_norm', 'two_norm', 'taxi_norm', 'inv_norm', 'state_str'],
                    data=state_data)
    df.to_csv(path_or_buf='state_data_'+str(SIDE)+'.csv', index=False)
    print('Done exploring, data written.\n')

explore_and_record()