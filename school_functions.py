import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools
from sympy.utilities.iterables import multiset_permutations
import operator, random
import simpy
import time
import json
from itertools import chain
import gurobipy as gp
from gurobipy import GRB

# Recovery Simulations
class Region(object):
    """A region has a limited number of contractors (``NUM_CONTRACTOR``) to
    construct buildings in parallel.
    
    Buildings have to request a contractor to build their house. When they
    get one, they can start the rebuilding process and wait for it to finish
    (which takes 'cons_time' days)

    """
    def __init__(self, env, num_contractor):
        self.env = env
        self.contractor = simpy.Resource(env, num_contractor)

    def rebuild(self, building, cons_time):
        """The rebuilding process. It takes a ``building`` process and tries
        to rebuild it"""
        yield self.env.timeout(cons_time)
        

def building(env, bldg_id, rg, damage, cons_time, data):
    """The building (each building has a ``name`` and 
    damage level 'damage') arrives at the region (``rg``) 
    and requests a rebuild.
    
    It then starts the rebuilding process, which takes a cons_time
    that is lognormally distributed. waits for it to finish and
    is reconstructured. 

    """

    # Construction
    with rg.contractor.request() as request:
        yield request

        # Construction
#         print('Building %d with damage level %s starts construction at %.2f.' % (bldg_id, str(damage), env.now))
        start_cons_time = env.now

        yield env.process(rg.rebuild(bldg_id, cons_time))

#         print('Building %d with damage level %s finishes construction at %.2f took %.2f days' % (bldg_id,str(damage), env.now, cons_time))

        # Append data of construction times
        data.append((bldg_id, start_cons_time, env.now))
#         data.append((bldg_id, env.now))

def setup(env, num_contractor, bldg_id, damage_building, cons_time, data):
    """Create a region and number of damaged buildings"""
    
    # Create the region
    region = Region(env, num_contractor)

    # Create buildings initially
    for i in range(len(damage_building)):
        env.process(building(env, bldg_id[i], region, 
                             damage_building[i], cons_time[i], data))
        yield env.timeout(0)

def simulate_recovery(num_contractor, ds, cons_time, order, sim_time = 5000):
    # Setup and start the simulation
    data_comp = []
    random.seed(0)  # This helps reproduce the results

    # Create an environment and start the setup process
    env = simpy.Environment()
    env.process(setup(env, num_contractor, order, ds[order], cons_time[order], data_comp))

    # Execute!
    env.run(until=sim_time)
    
#     print(data_comp)
    
    cons_order = [result[0] for result in data_comp]
    recov_time = np.array([0] + [result[-1] for result in data_comp])
    
    return data_comp, cons_order, recov_time

def get_distance(school_a, school_b, distance_df):
    '''
    Gets the distance between a pair of schools a and b from the distance matrix.
    Inputs:
    school_a and school_b: school IDs
    distance_df: distance dataframe
    
    Output: distance between school_a and school_b
    '''
    
    row = distance_df.loc[(distance_df['origin_id'] == school_a) & (distance_df['destination_id'] == school_b)]
    
    return row['total_cost'].values[0]

def nearest_neighbor_distance(school_idx, ds, distance_matrix):
    '''
    Gets the distance of the damaged school to the nearest functional school
    school_idx: Index of the school
    '''
    undamaged_schools = np.where(ds == 0)[0]
    
    return min([distance_matrix[school_idx][i] for i in undamaged_schools])

def weighted_demand(schools, ds, demand, distance_matrix):
    '''
    Calculates the weighted-demand given the functional schools. 
    Weighted-demand defined by p-median
    '''
    total = 0
    nearest_dist = [nearest_neighbor_distance(i,ds, distance_matrix) for i in range(len(schools))]
    total = sum([a*b for a, b in zip(nearest_dist, demand)])
        
    return nearest_dist, total

def weighted_demand_recov(schools,ds, order, demand, distance_matrix):
    '''
    Calculates the weighted-demand (p-median) over time as the schools are re-opened
    '''
    
    total = np.zeros(len(order)+1)
    nearest_dist= []
    tmp_result = weighted_demand(schools,ds, demand, distance_matrix)
    total[0] = tmp_result[-1]
    nearest_dist.append(tmp_result[0])
    ds_new = ds.copy()
    
    for i in range(len(order)):
        ds_new[order[i]] = 0
        tmp_result = weighted_demand(schools, ds_new, demand, distance_matrix)
        total[i+1] = tmp_result[-1]
        nearest_dist.append(tmp_result[0])

    return nearest_dist, total

def calculate_recov_time(cons_time, ds, order):
    '''
    Calculates the recovery curve (x-axis) depending on the order of reconstruction
    '''
    
    recov_order = cons_time[order]
    recov_time = np.cumsum(recov_order)
    x = np.concatenate(([0],recov_time))
    return x

def calculate_WD_area(recov_time, WD_total):
    '''
    Calculates the area under the curve
    '''
    return np.trapz(WD_total, recov_time)


def compute_cost(recov_time, total):
	'''
	computes the area under the curve (for comparing with the relaxed LP). 
	Instead of trapz, area under the curve is a step function. 
	Input: 
	recov_time: an array of the times a building is completed (the x-axis)
	total: an array of the weighted-demand cost every time a building is completed (y-axis)
	return: area under the curve
	'''

	x = recov_time[1:]
	y = total[:-1]
	cost = x[0]*y[0]
	for i in range(1, len(y)):
		cost_curr = y[i]*(x[i]- x[i-1])
		cost += cost_curr

	return cost


def compute_social_cost(schools, cons_time, ds, order, distances, demand):
    '''
    Compute the area under the curve (for comparing with the relaxed LP)
    '''
    # get initial values
    cost = 0
    repaired = np.where(ds == 0)[0]
    dmg_initial = np.nonzero(ds)[0]
    
    
    closest_nbr = np.empty(len(schools))
    smallest_dst = np.empty(len(schools))
    for i in range(len(dmg_initial)):
        opt_dst = np.Infinity
        argmin = -1
        for j in range(len(repaired)):
            dst = distances[dmg_initial[i]][repaired[j]]
            if dst < opt_dst:
                opt_dst = dst
                argmin = repaired[j]
        closest_nbr[dmg_initial[i]] = argmin
        smallest_dst[dmg_initial[i]] = opt_dst
    
    wd_curr = 0
    for i in range(len(dmg_initial)):
        wd_curr += smallest_dst[dmg_initial[i]] * demand[dmg_initial[i]]
    
    ds_new = ds.copy()
    
    for t in range(len(order)):
        # make the next in order repaired
        ds_new[order[t]] = 0
        
        # add to the cost 
        cost += wd_curr * cons_time[order[t]]
        
        # Update closest neighbors
        repaired = np.where(ds_new == 0)[0]
        dmg_curr = np.nonzero(ds_new)[0]
        for k in range(len(dmg_curr)):
            if smallest_dst[dmg_curr[k]] > distances[dmg_curr[k]][order[t]]:
                smallest_dst[dmg_curr[k]] = distances[dmg_curr[k]][order[t]]
                closest_nbr[dmg_curr[k]] = order[t]
        
        wd_curr = 0
        
        for k in range(len(dmg_curr)):
            wd_curr += smallest_dst[dmg_curr[k]] * demand[dmg_curr[k]]
        
    return cost

def greedy_alg(schools, ds, demand, cons_time, distances, max_moves = 10):
    '''
    schools: array of all schools in region
    ds: array of damage states for all schools in region (0,1,2,3)
    demand = array of number of students enrolled for each school
    cons_time = array of construction time for all schools (0 if undamaged)
    distances: a list of list for distance between pairs of schools in schools. 
    max_moves: the maximum number of moves students can undergo
    '''

    # get initial values
    repaired = np.where(ds == 0)[0]
    dmg_initial = np.nonzero(ds)[0]
    
    closest_nbr = np.zeros(len(schools))
    smallest_dst = np.zeros(len(schools))
    num_moves = np.zeros(len(schools))
    
    for i in range(len(dmg_initial)):
        opt_dst = np.Infinity
        argmin = -1
        for j in range(len(repaired)):
            dst = distances[dmg_initial[i]][repaired[j]]
            if dst < opt_dst:
                opt_dst = dst
                argmin = repaired[j]
        closest_nbr[dmg_initial[i]] = argmin
        smallest_dst[dmg_initial[i]] = opt_dst

        
    # update number of moves
    num_moves += (smallest_dst > 0)
    
    wd_curr = 0
    for i in range(len(dmg_initial)):
        wd_curr += smallest_dst[dmg_initial[i]] * demand[dmg_initial[i]]

    ds_new = ds.copy()
    order = []

    for j in range(len(dmg_initial)):
        dmg_idx = np.nonzero(ds_new)[0]
        gittins = np.zeros(len(dmg_idx))
        wd_next = np.zeros(len(dmg_idx))

        for i in range(len(dmg_idx)): # loop through each damaged school
            ds_tmp = ds_new.copy()
            ds_tmp[dmg_idx[i]] = 0 # assume that school i is reconstructed
            
            # Recalculate smallest nbrs 
            cl_nbr = np.copy(closest_nbr)
            sm_dst = np.copy(smallest_dst)
            
            wd_next[i] = 0

            for k in range(len(dmg_idx)):
                if i == k: # the students can go to original school
                    if (sm_dst[dmg_idx[k]] > distances[dmg_idx[k]][dmg_idx[i]]) and (num_moves[dmg_idx[k]] <= max_moves):
                        sm_dst[dmg_idx[k]] = distances[dmg_idx[k]][dmg_idx[i]]
                        cl_nbr[dmg_idx[k]] = dmg_idx[i]
                else:
                    if (sm_dst[dmg_idx[k]] > distances[dmg_idx[k]][dmg_idx[i]]) and (num_moves[dmg_idx[k]] < max_moves):
                        sm_dst[dmg_idx[k]] = distances[dmg_idx[k]][dmg_idx[i]]
                        cl_nbr[dmg_idx[k]] = dmg_idx[i]
                wd_next[i] += sm_dst[dmg_idx[k]] * demand[dmg_idx[k]]
            
            # Calculate gittin index
            gittins[i] = (wd_curr-wd_next[i])/cons_time[dmg_idx[i]]
            

        # Choose maximum gittins value
        next_cons = np.argmax(gittins)
        order.append(dmg_idx[next_cons])

        # Update values for next loop
        ds_new[dmg_idx[next_cons]] = 0
        wd_curr = wd_next[next_cons]
        
        # Update closest neighbors
        repaired = np.where(ds_new == 0)[0]
        dmg_curr = np.nonzero(ds_new)[0]
        
        prev_closest_nbr = closest_nbr.copy()
        
        for k in range(len(dmg_curr)):
            if (smallest_dst[dmg_curr[k]] > distances[dmg_curr[k]][dmg_idx[next_cons]]) and (num_moves[dmg_curr[k]] < max_moves):
                smallest_dst[dmg_curr[k]] = distances[dmg_curr[k]][dmg_idx[next_cons]]
                closest_nbr[dmg_curr[k]] = dmg_idx[next_cons]
                
        # for students that can go back to original school
        closest_nbr[dmg_idx[next_cons]] = 0
        smallest_dst[dmg_idx[next_cons]] = 0
                
        num_moves[dmg_curr] += (prev_closest_nbr[dmg_curr] != closest_nbr[dmg_curr])

    return num_moves, order


def lookahead_greedy(schools, ds, demand, cons_time, distance_matrix, num_contractor, sim_time):
    '''
    Modification of freedy algorithm for reconstruction prioritization of schools.
    Input:
    schools: a list of the school IDs
    ds: a list of the damage level (0-3) corresponding to each school
    demand: a list of the studnets enrolled at each school
    distance_matrix: matrix of the distance between pairwise schools
    
    Output:
    order: a list the optimal solution for reconstruction using modified greedy algorithm. 
           The numbers correspond to the index from the list of schools. 
    '''
    
    
    # get initial values
    dmg_initial = np.nonzero(ds)[0]
    _, wd_curr = weighted_demand(schools,ds, demand, distance_matrix)
    
    ds_new = ds.copy()
    order = []
    
    for j in range(len(dmg_initial)):
        dmg_idx = np.nonzero(ds_new)[0]
        best_ind = -1
        bestval = np.Infinity
        '''
        simul_values = np.empty(len(schools))
        for k in range(len(schools)):
            simul_values[k] = np.Infinity
        '''

        for i in range(len(dmg_idx)): # loop through each damaged school
            ds_tmp = ds_new.copy()
            ds_tmp[dmg_idx[i]] = 0 # assume that school is reconstructed
            order_so_far = order.copy()
            order_so_far.append(dmg_idx[i])
            
            # we now simulate Greedy until the end 
            _, continuation = greedy_alg(schools, ds_tmp, demand, cons_time, distance_matrix)
            #print("over")
            for r in range(len(continuation)):
                order_so_far.append(continuation[r])
            

            # Calculate result of simulation
            _, cons_order, recov_time = simulate_recovery(num_contractor, ds, cons_time, order_so_far, sim_time)
            _, total = weighted_demand_recov(schools, ds, cons_order, demand, distance_matrix)
            area = calculate_WD_area(recov_time, total)
            #simul_values[dmg_idx[i]] = area
            if area < bestval:
                bestval = area
                best_ind = i
#             print(area)

        # Choose maximum gittins value
        #print(simul_values[dmg_idx])
        next_cons = best_ind
        #print(simul_values[next_cons])
        order.append(dmg_idx[next_cons])

        # Update values for next loop
        ds_new[dmg_idx[next_cons]] = 0
        _, wd_curr = weighted_demand(schools,ds_new, demand, distance_matrix)
        
#         print(dmg_idx[next_cons])
        
    return order


def two_step_lookahead(schools, ds, demand, cons_time, distance_matrix, num_contractor, sim_time):
    # get initial values
    dmg_initial = np.nonzero(ds)[0]
    _, wd_curr = weighted_demand(schools,ds, demand, distance_matrix)
    
    ds_new = ds.copy()
    order = []
    
    for j in range(len(dmg_initial)):
        dmg_idx = np.nonzero(ds_new)[0]
        best_ind = -1
        bestval = np.Infinity
        '''
        simul_values = np.empty(len(schools))
        for k in range(len(schools)):
            simul_values[k] = np.Infinity
        '''

        for i in range(len(dmg_idx)): # loop through each damaged school
            ds_tmp = ds_new.copy()
            ds_tmp[dmg_idx[i]] = 0 # assume that school is reconstructed
            order_so_far = order.copy()
            order_so_far.append(dmg_idx[i])
            
            # we now simulate one-step-improvement Greedy until the end 
            continuation = lookahead_greedy(schools, ds_tmp, demand, cons_time, distance_matrix, num_contractor, sim_time)
#             print("over")
            for r in range(len(continuation)):
                order_so_far.append(continuation[r])
            

            # Calculate result of simulation
            _, cons_order, recov_time = simulate_recovery(num_contractor, ds, cons_time, order_so_far, sim_time)
            _, total = weighted_demand_recov(schools, ds, cons_order, demand, distance_matrix)
            area = calculate_WD_area(recov_time, total)
            
            if area < bestval:
                bestval = area
                best_ind = i
            

        # Choose maximum gittins value
        next_cons = best_ind
        order.append(dmg_idx[next_cons])

        # Update values for next loop
        ds_new[dmg_idx[next_cons]] = 0
        _, wd_curr = weighted_demand(schools,ds_new, demand, distance_matrix)
        
#         print(dmg_idx[next_cons])
        
    return order

def three_step_lookahead(schools, ds, demand, cons_time, distance_matrix, num_contractor, sim_time):
    # get initial values
    dmg_initial = np.nonzero(ds)[0]
    _, wd_curr = weighted_demand(schools,ds, demand, distance_matrix)
    
    ds_new = ds.copy()
    order = []
    
    for j in range(len(dmg_initial)):
        dmg_idx = np.nonzero(ds_new)[0]
        best_ind = -1
        bestval = np.Infinity
        '''
        simul_values = np.empty(len(schools))
        for k in range(len(schools)):
            simul_values[k] = np.Infinity
        '''

        for i in range(len(dmg_idx)): # loop through each damaged school
            ds_tmp = ds_new.copy()
            ds_tmp[dmg_idx[i]] = 0 # assume that school is reconstructed
            order_so_far = order.copy()
            order_so_far.append(dmg_idx[i])
            
            # we now simulate one-step-improvement Greedy until the end 
            continuation = two_step_lookahead(schools, ds_tmp, demand, cons_time, distance_matrix)
#             print("over")
            for r in range(len(continuation)):
                order_so_far.append(continuation[r])
            

            # Calculate result of simulation
            _, cons_order, recov_time = simulate_recovery(num_contractor, ds, cons_time, order_so_far, sim_time)
            _, total = weighted_demand_recov(schools, ds, cons_order, demand, distance_matrix)
            area = calculate_WD_area(recov_time, total)
            
            if area < bestval:
                bestval = area
                best_ind = i
            

        # Choose maximum gittins value
        next_cons = best_ind
        order.append(dmg_idx[next_cons])

        # Update values for next loop
        ds_new[dmg_idx[next_cons]] = 0
        _, wd_curr = weighted_demand(schools,ds_new, demand, distance_matrix)
        
#         print(dmg_idx[next_cons])
        
    return order

def gantt_modified(SCHEDULE, MACHINES):
    '''
    Input:
    SCHEDULE: a dictionary with key School ID and value a list containing dicts with keys: start, finish, and machine
    example:
    {3327: [{'start': 51, 'finish': 52, 'machine': 1},
            {'start': 52, 'finish': 53, 'machine': 1},
            {'start': 53, 'finish': 54, 'machine': 1}],
     3350: [{'start': 60, 'finish': 61, 'machine': 1}],
     3286: [{'start': 34, 'finish': 35, 'machine': 1},
            {'start': 43, 'finish': 44, 'machine': 1}]}
    MACHINES: a list of contractor IDs

    Output:
    A gantt chart of the construction schedule
    '''
    bw = 0.3
    plt.figure(figsize=(12, 0.7*(len(SCHEDULE.keys()))))
    idx = 0
    for j in SCHEDULE.keys():
        for k in range(len(SCHEDULE[j])):
            x = SCHEDULE[j][k]['start']
            y = SCHEDULE[j][k]['finish']
            plt.fill_between([x,y],[idx-bw,idx-bw],[idx+bw,idx+bw], color='red', alpha=0.5)
            plt.plot([x,y,y,x,x], [idx-bw,idx-bw,idx+bw,idx+bw,idx-bw],color='k')
            plt.text((SCHEDULE[j][k]['start'] + SCHEDULE[j][k]['finish'])/2.0,idx,
                str(SCHEDULE[j][k]['machine']), color='white', weight='bold',
                horizontalalignment='center', verticalalignment='center')
        idx += 1

    plt.ylim(-0.5, idx-0.5)
    plt.title('Construction Schedule')
    plt.xlabel('Time')
    plt.ylabel('Schools')
    plt.yticks(range(len(SCHEDULE)), SCHEDULE.keys())
    plt.grid()
    xlim = plt.xlim()

    plt.figure(figsize=(12, 0.7*len(MACHINES)))
    for j in SCHEDULE.keys():
        for k in range(len(SCHEDULE[j])):
            idx = MACHINES.index(SCHEDULE[j][k]['machine'])
            x = SCHEDULE[j][k]['start']
            y = SCHEDULE[j][k]['finish']
            plt.fill_between([x,y],[idx-bw,idx-bw],[idx+bw,idx+bw], color='red', alpha=0.5)
            plt.plot([x,y,y,x,x], [idx-bw,idx-bw,idx+bw,idx+bw,idx-bw],color='k')
            plt.text((SCHEDULE[j][k]['start'] + SCHEDULE[j][k]['finish'])/2.0,idx,
                str(j), color='white', weight='bold',
                horizontalalignment='center', verticalalignment='center')
    plt.xlim(xlim)
    plt.ylim(-0.5, len(MACHINES)-0.5)
    plt.title('Contractor Schedule')
    plt.yticks(range(len(MACHINES)), MACHINES)
    plt.ylabel('Contractors')
    plt.grid()
