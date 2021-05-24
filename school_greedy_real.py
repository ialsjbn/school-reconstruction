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
import copy
from gurobipy import GRB

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

def get_demand_order(schools_id, ds, demand_list):
	damaged_idx = np.where(ds != 0)[0]

	# Get ordering based on most enrolled
	elsort = np.argsort(demand_list)[::-1]
	order_demand = schools_id[[i for i in elsort if i in damaged_idx]]

	return order_demand

def get_time_demand_order(schools_id, ds, cons_time, demand_list):
	damaged_idx = np.where(ds != 0)[0]
	tuple_list = [(schools_id[i], cons_time[schools_id[i]], demand_list[i]) for i in damaged_idx]
	sorter = lambda x: (x[1], -x[2])
	sorted_list = sorted(tuple_list, key = sorter)
	time_demand_order = [tup[0] for tup in sorted_list]

	return time_demand_order

def get_damage_demand_order(schools_id, ds, demand_list):
	ds_damage_order = []
	ds_sort = np.argsort(ds)[::-1]
	for d in np.unique(ds)[:0:-1]: # for damage states
	    tmp = ds[ds_sort]
	    tmp2 = [ds_sort[i] for i in range(len(ds)) if tmp[i] == d]
	    tmp2 = np.array(tmp2)
	    tmp3 = demand_list[tmp2]
	    tmp4 = np.argsort(tmp3)[::-1].tolist()
	    ds_damage_order.append(tmp2[tmp4].tolist())  
	ds_damage_order = schools_id[[j for i in ds_damage_order for j in i]]

	return ds_damage_order


def get_nearest_school_order(original_school, distance_matrix, undamaged_schools):
    '''
    Given the original_school, outputs a list of functional schools ID starting from the closest to farthest
    '''
    temp_dict = {school_pair:dist for school_pair,dist in distance_matrix.items() if ((school_pair[0] == original_school) and 
                                                                                      (school_pair[1] in undamaged_schools))}
    return [s[0][1] for s in sorted(temp_dict.items(), key = lambda value: value[1])]

def calculate_potential(to_reconstruct, membership, moves_used, original, where, distance_matrix, total_students, classroom, B, M):
    '''
    Function to calculate the potential cost reduction of reconstructing a school
    
    Input:
    to_reconstruct: school ID of the potentially reconstructed school
    membership: dict with keys of School IDs, and values a list of the students in the school
    moves_used: a list len(total_students) indicating the number of moves the students have done
    where: a list len(total_students) indicating which school ID the students are currently at
    distance_matrix: a dict with keys school ID pairs, and value the distance between the two schools
    total_students: the number of students in region
    B: maximum student-to-classroom ratio
    M: maximum number of moves
    
    Output:
    member_curr: updated membership dict if school to_reconstruct was reconstructed
    moves_curr: updated moves_used list if school to_reconstruct was reconstructed
    where_curr: updated where list if school to_reconstruct was reconstructed
    potential: cost reduction if school to_reconstruct was reconstructed
    '''
    
    # makes sure original dicts and lists aren't modified
    member_curr = copy.deepcopy(membership)
    moves_curr = copy.deepcopy(moves_used)
    where_curr = copy.deepcopy(where)
    
    potential = 0
    dist_reduction = np.zeros(total_students)

    for s in range(total_students):
        dist_reduction[s] = max(0, distance_matrix[(original[s], where_curr[s])] - distance_matrix[(original[s], to_reconstruct)])
            
    # split students that can go back to original school i and not
    students_in_i = [s for s in range(total_students) if original[s] == to_reconstruct]
    students_not_in_i =  [s for s in range(total_students) if original[s] != to_reconstruct] #np.setdiff1d(list(range(total_students)),students_in_i)

    # prioritize students that can go back to original school
    for s in students_in_i: 
        potential += dist_reduction[s]
        member_curr[to_reconstruct].append(s)
        member_curr[where_curr[s]].remove(s)
        where_curr[s]= to_reconstruct

    # sort leftover students from highest distance reduction
    students_not_in_i_sorted = [student for student in np.argsort(-1*dist_reduction) if student in students_not_in_i]    
    
    for s in students_not_in_i_sorted:
        # move the student to the closer reconstructed school
        if dist_reduction[s] > 0 and (len(member_curr[to_reconstruct]) < classroom[to_reconstruct]*B) and moves_curr[s] < M : 
            member_curr[to_reconstruct].append(s)
            member_curr[where_curr[s]].remove(s)
            moves_curr[s] += 1
            potential += dist_reduction[s]
            where_curr[s] = to_reconstruct
        else:
            continue

    return member_curr, moves_curr, where_curr, potential

def greedy_capacity(schools_id, ds, demand, cons_time, classroom, distance_matrix, B = 1000, M= 10):
    '''
    Function to calculate school reconstruction ordering accounting for capacity and maximum number of moves
    
    Input:
    schools_id: array of school IDs in the region
    ds: array of damage states for all schools in the region (0,1,2,3)
    demand: dict with keys school ID and values number of students enrolled for each school
    cons_time: dict of reconstruction time with keys school ID and values the construction time in days
    classroom: dict with keys school ID and values the number of classroom in the corresponding school
    distance_matrix: a dict with keys school ID pairs, and value the distance between the two schools
    B: maximum student to classroom ratio
    M: maximum number of moves students can perform
    
    Output:
    order: array of school ID ordered based on the greedy algorithm
    cost: array of costs (weighted-demand)
    '''

    damaged_schools = schools_id[ds != 0]
    num_damaged = len(damaged_schools)
    undamaged_schools = schools_id[ds == 0]

    # total number of students in region
    total_students = sum(demand.values())

    # initial capacity with damaged schools
    initial_capacity = sum([value for key,value in classroom.items() if key in undamaged_schools])*B

    # initial check if problem is feasible
    if total_students > initial_capacity:
        raise ValueError('Not enough capacity!')

    # Initialize arrays    
    original = np.repeat(schools_id[0], demand[schools_id[0]]) # original schools student went
    for i in range(1, len(schools_id)):
        original = np.append(original, np.repeat(schools_id[i], demand[schools_id[i]])) 
    moves_used = np.zeros(total_students) # number of moves used by students
    where = original.copy()

    # add students to original school
    membership = {school: [] for school in schools_id} 
    for s in range(total_students):
        membership[original[s]].append(s)

    # allocate students of damaged schools to nearest functional school, making sure it doesn't exceed capacity
    cost = np.zeros(num_damaged+1)
    for s in range(total_students):
        if original[s] in damaged_schools: # only move students in damaged schools
            potential_school_move = get_nearest_school_order(original[s], distance_matrix, undamaged_schools)
            for i in range(len(potential_school_move)):
                if len(membership[potential_school_move[i]]) < classroom[potential_school_move[i]]*B: # there is still enough space
                    membership[potential_school_move[i]].append(s)
                    membership[original[s]].remove(s)
                    moves_used[s] += 1
                    where[s] = potential_school_move[i]
                    cost[0] += distance_matrix[(original[s], where[s])] # calculate initial cost
                    break

    order = []
    damaged_curr = schools_id[ds != 0].copy()
    cost_reduction = np.zeros(num_damaged)
    for d in range(num_damaged):# loop through damaged schools and reconstruct
    #     print('School {} out of {}'.format(d, num_damaged))
        potential = np.zeros(len(damaged_curr))

        # calculate gittins index for each damaged school
        index = 0
        for i in damaged_curr:
            _,_,_,potential[index] = calculate_potential(i, membership, moves_used, original, where, 
                                                   distance_matrix, total_students, classroom, B, M)
            index += 1

        # choose school with maximum potential
        chosen_school_index = np.argmax(potential/[cons_time[i] for i in damaged_curr])
        chosen_school = damaged_curr[chosen_school_index]
        order.append(chosen_school)

        # update all other values
        membership, moves_used, where, cost_reduction[d] = calculate_potential(chosen_school, membership, moves_used, original, where,  
                                                   distance_matrix, total_students, classroom, B, M)
        cost[d+1] = cost[d] - cost_reduction[d]
        damaged_curr = damaged_curr[damaged_curr != chosen_school] # chosen school no longer damaged
        undamaged_schools = np.append(undamaged_schools, chosen_school) # chosen school is now reconstructed
        
    return order, cost, cost_reduction

def naive_capacity(order, schools_id, ds, demand, classroom, distance_matrix, B = 1000, M = 10):
    '''
    Function to calculate cost given order accounting for capacity and maximum number of moves
    
    Input:
    order: array of school IDs with given naive order
    schools_id: array of school IDs in the region
    ds: array of damage states for all schools in the region (0,1,2,3)
    demand: dict with keys school ID and values number of students enrolled for each school
    classroom: dict with keys school ID and values the number of classroom in the corresponding school
    distance_matrix: a dict with keys school ID pairs, and value the distance between the two schools
    B: maximum student to classroom ratio
    M: maximum number of moves students can perform
    
    Output:
    cost: array of costs (weighted-demand)
    '''

    damaged_schools = schools_id[ds != 0]
    num_damaged = len(damaged_schools)
    undamaged_schools = schools_id[ds == 0]

    # total number of students in region
    total_students = sum(demand.values())

    # initial capacity with damaged schools
    initial_capacity = sum([value for key,value in classroom.items() if key in undamaged_schools])*B

    # initial check if problem is feasible
    if total_students > initial_capacity:
        raise ValueError('Not enough capacity!')

    # Initialize arrays    
    original = np.repeat(schools_id[0], demand[schools_id[0]]) # original schools student went
    for i in range(1, len(schools_id)):
        original = np.append(original, np.repeat(schools_id[i], demand[schools_id[i]])) 
    moves_used = np.zeros(total_students) # number of moves used by students
    where = original.copy()

    # add students to original school
    membership = {school: [] for school in schools_id} 
    for s in range(total_students):
        membership[original[s]].append(s)

    # allocate students of damaged schools to nearest functional school, making sure it doesn't exceed capacity
    cost = np.zeros(num_damaged+1)
    for s in range(total_students):
        if original[s] in damaged_schools: # only move students in damaged schools
            potential_school_move = get_nearest_school_order(original[s], distance_matrix, undamaged_schools)
            for i in range(len(potential_school_move)):
                if len(membership[potential_school_move[i]]) < classroom[potential_school_move[i]]*B: # there is still enough space
                    membership[potential_school_move[i]].append(s)
                    membership[original[s]].remove(s)
                    moves_used[s] += 1
                    where[s] = potential_school_move[i]
                    cost[0] += distance_matrix[(original[s], where[s])] # calculate initial cost
                    break

    damaged_curr = schools_id[ds != 0].copy()

    for d in range(len(order)):# loop through damaged schools and reconstruct
        # update all other values
        membership, moves_used, where, cost_reduction = calculate_potential(order[d], membership, moves_used, original, where,  
                                                   distance_matrix, total_students, classroom, B, M)
        cost[d+1] = cost[d] - cost_reduction
        damaged_curr = damaged_curr[damaged_curr != order[d]] # chosen school no longer damaged
        undamaged_schools = np.append(undamaged_schools, order[d]) # chosen school is now reconstructed
        
    return cost

def calculate_potential_tmp(to_reconstruct, tmpschool_id, membership, moves_used, original, where, distance_matrix, total_students, classroom, B, M, P):
    '''
    Function to calculate the potential cost reduction of reconstructing a school
    Realistic case with temporary schools
    
    Input:
    to_reconstruct: school ID of the potentially reconstructed school
    tmpschool_id: array of temporary school IDs in the region
    membership: dict with keys of School IDs, and values a list of the students in the school
    moves_used: a list len(total_students) indicating the number of moves the students have done
    where: a list len(total_students) indicating which school ID the students are currently at
    distance_matrix: a dict with keys school ID pairs, and value the distance between the two schools
    total_students: the number of students in region
    B: maximum student-to-classroom ratio
    M: maximum number of moves
    
    Output:
    member_curr: updated membership dict if school to_reconstruct was reconstructed
    moves_curr: updated moves_used list if school to_reconstruct was reconstructed
    where_curr: updated where list if school to_reconstruct was reconstructed
    potential: cost reduction if school to_reconstruct was reconstructed
    '''
    
    # makes sure original dicts and lists aren't modified
    member_curr = copy.deepcopy(membership)
    moves_curr = copy.deepcopy(moves_used)
    where_curr = copy.deepcopy(where)
    
    potential = 0
    dist_reduction = np.zeros(total_students)

    for s in range(total_students):
        if where_curr[s] in tmpschool_id: # student is currently in tmp school
            dist_reduction[s] = max(0, distance_matrix[(original[s], where_curr[s])] + P - distance_matrix[(original[s], to_reconstruct)])
        else: # student in functional school
            dist_reduction[s] = max(0, distance_matrix[(original[s], where_curr[s])] - distance_matrix[(original[s], to_reconstruct)])
            
    # split students that can go back to original school i and not
    students_in_i = [s for s in range(total_students) if original[s] == to_reconstruct]
    students_not_in_i =  [s for s in range(total_students) if original[s] != to_reconstruct] #np.setdiff1d(list(range(total_students)),students_in_i)

    # prioritize students that can go back to original school
    for s in students_in_i: 
        potential += dist_reduction[s]
        member_curr[to_reconstruct].append(s)
        member_curr[where_curr[s]].remove(s)
        where_curr[s]= to_reconstruct

    # sort leftover students from highest distance reduction
    students_not_in_i_sorted = [student for student in np.argsort(-1*dist_reduction) if student in students_not_in_i]    
    
    for s in students_not_in_i_sorted:
        # move the student to the closer reconstructed school
        if dist_reduction[s] > 0 and (len(member_curr[to_reconstruct]) < classroom[to_reconstruct]*B) and moves_curr[s] < M : 
            member_curr[to_reconstruct].append(s)
            member_curr[where_curr[s]].remove(s)
            moves_curr[s] += 1
            potential += dist_reduction[s]
            where_curr[s] = to_reconstruct
        else:
            continue

    return member_curr, moves_curr, where_curr, potential

def greedy_capacity_tmp(schools_id, tmpschool_id, ds, demand, cons_time, classroom, distance_matrix, B, M, P):
    '''
    Function to calculate school reconstruction ordering accounting for capacity, 
    maximum number of moves, and temporary schools
    
    Input:
    schools_id: array of school IDs in the region
    tmpschool_id: array of temporary school IDs in the region
    ds: array of damage states for all schools in the region (0,1,2,3)
    demand: dict with keys school ID and values number of students enrolled for each school (includes temp school)
    cons_time: dict of reconstruction time with keys school ID and values the construction time in days
    classroom: dict with keys school ID and values the number of classroom in the corresponding school (includes temp school)
    distance_matrix: a dict with keys school ID pairs, and value the distance between the two schools (includes temp school)
    B: maximum student to classroom ratio
    M: maximum number of moves students can perform
    P: penalty for being in the temporary school
    
    Output:
    order: array of school ID ordered based on the greedy algorithm
    cost: array of costs (weighted-demand)
    last_temp: a dict with of the last days students are in each temp school (key is temp school ID, value is day)
    cost_reduction: the cost reduction at each time step
    '''

    damaged_schools = schools_id[ds != 0]
    num_damaged = len(damaged_schools)
    undamaged_schools = schools_id[ds == 0]
    functional_schools = np.append(undamaged_schools, tmpschool_id) # undamaged + temporary schools

    # total number of students in region
    total_students = sum(demand.values())

    # initial capacity with damaged schools
    initial_capacity = sum([value for key,value in classroom.items() if key in functional_schools])*B

    # initial check if problem is feasible
    if total_students > initial_capacity:
        raise ValueError('Not enough capacity!')

    # Initialize arrays    
    original = np.repeat(schools_id[0], demand[schools_id[0]]) # original schools student went
    for i in range(1, len(schools_id)):
        original = np.append(original, np.repeat(schools_id[i], demand[schools_id[i]])) 
    moves_used = np.zeros(total_students) # number of moves used by students
    where = original.copy()

    # add students to original school
    membership = {school: [] for school in schools_id} 
    for s in range(total_students):
        membership[original[s]].append(s)
    for i in tmpschool_id: # initialize temporary school membership
        membership[i] = []

    # allocate students of damaged schools to nearest functional school, making sure it doesn't exceed capacity
    cost = np.zeros(num_damaged+1)
    for s in range(total_students):
        if original[s] in damaged_schools: # only move students in damaged schools
            potential_school_move = get_nearest_school_order(original[s], distance_matrix, functional_schools)
            for i in range(len(potential_school_move)):
                if len(membership[potential_school_move[i]]) < classroom[potential_school_move[i]]*B: # there is still enough space
                    membership[potential_school_move[i]].append(s)
                    membership[original[s]].remove(s)
                    moves_used[s] += 1
                    where[s] = potential_school_move[i]
                    cost[0] += distance_matrix[(original[s], where[s])] # calculate initial cost
                    if where[s] in tmpschool_id: # add penalty if student in temporary school
                        cost[0] += P
                    break

    order = []
    last_temp = {}
    damaged_curr = schools_id[ds != 0].copy()
    cost_reduction = np.zeros(num_damaged)
    
    for d in range(num_damaged):# loop through damaged schools and reconstruct
    #     print('School {} out of {}'.format(d, num_damaged))
        potential = np.zeros(len(damaged_curr))

        # calculate gittins index for each damaged school
        index = 0
        for i in damaged_curr:
            _,_,_,potential[index] = calculate_potential_tmp(i, tmpschool_id, membership, moves_used, original, where, 
                                                   distance_matrix, total_students, classroom, B, M, P)
            index += 1

        # choose school with maximum potential
        chosen_school_index = np.argmax(potential/[cons_time[i] for i in damaged_curr])
        chosen_school = damaged_curr[chosen_school_index]
        order.append(chosen_school)

        # update all other values
        membership, moves_used, where, cost_reduction[d] = calculate_potential_tmp(chosen_school, tmpschool_id, membership, moves_used, original, 
                                                                            where, distance_matrix, total_students, classroom, B, M, P)
        cost[d+1] = cost[d] - cost_reduction[d]
        damaged_curr = damaged_curr[damaged_curr != chosen_school] # chosen school no longer damaged
        undamaged_schools = np.append(undamaged_schools, chosen_school) # chosen school is now reconstructed
        
        # keep track of the last time period students are in each temporary school
        for i in tmpschool_id:
            if not membership[i] and not i in last_temp.keys(): # no more students in the temporary school
                last_temp[i] = sum([cons_time[j] for j in order])
        
    return order, cost, last_temp, cost_reduction

def naive_capacity_tmp(order, schools_id, tmpschool_id, ds, demand, classroom, distance_matrix, B, M, P):
    '''
    Function to calculate cost given order accounting for capacity and maximum number of moves
    Realistic case with temporary schools
    
    Input:
    order: array of school IDs with given naive order
    schools_id: array of school IDs in the region
    tmpschool_id: array of temporary school IDs in the region
    ds: array of damage states for all schools in the region (0,1,2,3)
    demand: dict with keys school ID and values number of students enrolled for each school
    classroom: dict with keys school ID and values the number of classroom in the corresponding school
    distance_matrix: a dict with keys school ID pairs, and value the distance between the two schools
    B: maximum student to classroom ratio
    M: maximum number of moves students can perform
    P: penalty for being in the temporary school
    
    Output:
    cost: array of costs (weighted-demand)
    '''

    damaged_schools = schools_id[ds != 0]
    num_damaged = len(damaged_schools)
    undamaged_schools = schools_id[ds == 0]
    functional_schools = np.append(undamaged_schools, tmpschool_id) # undamaged + temporary schools

    # total number of students in region
    total_students = sum(demand.values())

    # initial capacity with damaged schools
    initial_capacity = sum([value for key,value in classroom.items() if key in functional_schools])*B

    # initial check if problem is feasible
    if total_students > initial_capacity:
        raise ValueError('Not enough capacity!')

    # Initialize arrays    
    original = np.repeat(schools_id[0], demand[schools_id[0]]) # original schools student went
    for i in range(1, len(schools_id)):
        original = np.append(original, np.repeat(schools_id[i], demand[schools_id[i]])) 
    moves_used = np.zeros(total_students) # number of moves used by students
    where = original.copy()

    # add students to original school
    membership = {school: [] for school in schools_id} 
    for s in range(total_students):
        membership[original[s]].append(s)
    for i in tmpschool_id: # initialize temporary school membership
        membership[i] = []

    # allocate students of damaged schools to nearest functional school, making sure it doesn't exceed capacity
    cost = np.zeros(num_damaged+1)
    for s in range(total_students):
        if original[s] in damaged_schools: # only move students in damaged schools
            potential_school_move = get_nearest_school_order(original[s], distance_matrix, functional_schools)
            for i in range(len(potential_school_move)):
                if len(membership[potential_school_move[i]]) < classroom[potential_school_move[i]]*B: # there is still enough space
                    membership[potential_school_move[i]].append(s)
                    membership[original[s]].remove(s)
                    moves_used[s] += 1
                    where[s] = potential_school_move[i]
                    cost[0] += distance_matrix[(original[s], where[s])] # calculate initial cost
                    if where[s] in tmpschool_id: # add penalty if student in temporary school
                        cost[0] += P
                    break

    damaged_curr = schools_id[ds != 0].copy()

    for d in range(len(order)):# loop through damaged schools and reconstruct
        # update all other values
        membership, moves_used, where, cost_reduction = calculate_potential_tmp(order[d], tmpschool_id, membership, moves_used, original, where,  
                                                   distance_matrix, total_students, classroom, B, M, P)
        cost[d+1] = cost[d] - cost_reduction
        damaged_curr = damaged_curr[damaged_curr != order[d]] # chosen school no longer damaged
        undamaged_schools = np.append(undamaged_schools, order[d]) # chosen school is now reconstructed
        
    return cost

def greedy_capacity_diff(schools_id, tmpschool_id, ds, demand, cons_time, classroom, distance_matrix, B, M, P):
    '''
    Function to calculate school reconstruction ordering accounting for capacity, 
    maximum number of moves, and temporary schools. 
    Calculations the cost disaggregated by schools
    
    Input:
    schools_id: array of school IDs in the region
    tmpschool_id: array of temporary school IDs in the region
    ds: array of damage states for all schools in the region (0,1,2,3)
    demand: dict with keys school ID and values number of students enrolled for each school (includes temp school)
    cons_time: dict of reconstruction time with keys school ID and values the construction time in days
    classroom: dict with keys school ID and values the number of classroom in the corresponding school (includes temp school)
    distance_matrix: a dict with keys school ID pairs, and value the distance between the two schools (includes temp school)
    B: maximum student to classroom ratio
    M: maximum number of moves students can perform
    P: penalty for being in the temporary school
    
    Output:
    order: array of school ID ordered based on the greedy algorithm
    cost: array of costs (weighted-demand)
    cost_reduction: 
    cost_school: array of costs per school
    '''

    damaged_schools = schools_id[ds != 0]
    num_damaged = len(damaged_schools)
    undamaged_schools = schools_id[ds == 0]
    functional_schools = np.append(undamaged_schools, tmpschool_id) # undamaged + temporary schools

    # total number of students in region
    total_students = sum(demand.values())

    # initial capacity with damaged schools
    initial_capacity = sum([value for key,value in classroom.items() if key in functional_schools])*B

    # initial check if problem is feasible
    if total_students > initial_capacity:
        raise ValueError('Not enough capacity!')

    # Initialize arrays    
    original = np.repeat(schools_id[0], demand[schools_id[0]]) # original schools student went
    for i in range(1, len(schools_id)):
        original = np.append(original, np.repeat(schools_id[i], demand[schools_id[i]])) 
    moves_used = np.zeros(total_students) # number of moves used by students
    where = original.copy()

    # add students to original school
    membership = {school: [] for school in schools_id} 
    for s in range(total_students):
        membership[original[s]].append(s)
    for i in tmpschool_id: # initialize temporary school membership
        membership[i] = []

    # allocate students of damaged schools to nearest functional school, making sure it doesn't exceed capacity
    cost = np.zeros(num_damaged+1)
    cost_school = {school: np.zeros(num_damaged+1) for school in damaged_schools} # keep track of cost for each school
    
    for s in range(total_students):
        if original[s] in damaged_schools: # only move students in damaged schools
            potential_school_move = get_nearest_school_order(original[s], distance_matrix, functional_schools)
            for i in range(len(potential_school_move)):
                if len(membership[potential_school_move[i]]) < classroom[potential_school_move[i]]*B: # there is still enough space
                    membership[potential_school_move[i]].append(s)
                    membership[original[s]].remove(s)
                    moves_used[s] += 1
                    where[s] = potential_school_move[i]
                    cost[0] += distance_matrix[(original[s], where[s])] # calculate initial cost
                    cost_school[original[s]][0] += distance_matrix[(original[s], where[s])]
                    if where[s] in tmpschool_id: # add penalty if student in temporary school
                        cost[0] += P
                        cost_school[original[s]][0] += P
                    break

    order = []
    last_temp = {}
    damaged_curr = schools_id[ds != 0].copy()
    cost_reduction = np.zeros(num_damaged)
    
    for d in range(num_damaged):# loop through damaged schools and reconstruct
    #     print('School {} out of {}'.format(d, num_damaged))
        potential = np.zeros(len(damaged_curr))

        # calculate gittins index for each damaged school
        index = 0
        for i in damaged_curr:
            _,_,_,potential[index] = calculate_potential_tmp(i, tmpschool_id, membership, moves_used, original, where, 
                                                   distance_matrix, total_students, classroom, B, M, P)
            index += 1

        # choose school with maximum potential
        chosen_school_index = np.argmax(potential/[cons_time[i] for i in damaged_curr])
        chosen_school = damaged_curr[chosen_school_index]
        order.append(chosen_school)

        # update all other values
        membership, moves_used, where, cost_reduction[d] = calculate_potential_tmp(chosen_school, tmpschool_id, membership, moves_used, original, 
                                                                            where, distance_matrix, total_students, classroom, B, M, P)
        cost[d+1] = cost[d] - cost_reduction[d]
        
        for s in range(total_students):
            if original[s] in damaged_schools:
                cost_school[original[s]][d+1] += distance_matrix[(original[s], where[s])] 
                if where[s] in tmpschool_id:
                    cost_school[original[s]][d+1] += P
        
        damaged_curr = damaged_curr[damaged_curr != chosen_school] # chosen school no longer damaged
        undamaged_schools = np.append(undamaged_schools, chosen_school) # chosen school is now reconstructed
        
        # keep track of the last time period students are in each temporary school
        for i in tmpschool_id:
            if not membership[i] and not i in last_temp.keys(): # no more students in the temporary school
                last_temp[i] = sum([cons_time[j] for j in order])
        
    return order, cost, last_temp, cost_reduction, cost_school

def naive_capacity_diff(order, schools_id, tmpschool_id, ds, demand, classroom, distance_matrix, B, M, P):
    '''
    Function to calculate cost given order accounting for capacity and maximum number of moves
    Calculates the cost disaggregated by schools
    
    Input:
    order: array of school IDs with given naive order
    schools_id: array of school IDs in the region
    tmpschool_id: array of temporary school IDs in the region
    ds: array of damage states for all schools in the region (0,1,2,3)
    demand: dict with keys school ID and values number of students enrolled for each school
    classroom: dict with keys school ID and values the number of classroom in the corresponding school
    distance_matrix: a dict with keys school ID pairs, and value the distance between the two schools
    B: maximum student to classroom ratio
    M: maximum number of moves students can perform
    P: penalty for being in the temporary school
    
    Output:
    cost: array of costs (weighted-demand)
    cost_school: array of costs per school
    '''

    damaged_schools = schools_id[ds != 0]
    num_damaged = len(damaged_schools)
    undamaged_schools = schools_id[ds == 0]
    functional_schools = np.append(undamaged_schools, tmpschool_id) # undamaged + temporary schools

    # total number of students in region
    total_students = sum(demand.values())

    # initial capacity with damaged schools
    initial_capacity = sum([value for key,value in classroom.items() if key in functional_schools])*B

    # initial check if problem is feasible
    if total_students > initial_capacity:
        raise ValueError('Not enough capacity!')

    # Initialize arrays    
    original = np.repeat(schools_id[0], demand[schools_id[0]]) # original schools student went
    for i in range(1, len(schools_id)):
        original = np.append(original, np.repeat(schools_id[i], demand[schools_id[i]])) 
    moves_used = np.zeros(total_students) # number of moves used by students
    where = original.copy()

    # add students to original school
    membership = {school: [] for school in schools_id} 
    for s in range(total_students):
        membership[original[s]].append(s)
    for i in tmpschool_id: # initialize temporary school membership
        membership[i] = []

    # allocate students of damaged schools to nearest functional school, making sure it doesn't exceed capacity
    cost = np.zeros(num_damaged+1)
    cost_school = {school: np.zeros(num_damaged+1) for school in damaged_schools} # keep track of cost for each school
    
    for s in range(total_students):
        if original[s] in damaged_schools: # only move students in damaged schools
            potential_school_move = get_nearest_school_order(original[s], distance_matrix, functional_schools)
            for i in range(len(potential_school_move)):
                if len(membership[potential_school_move[i]]) < classroom[potential_school_move[i]]*B: # there is still enough space
                    membership[potential_school_move[i]].append(s)
                    membership[original[s]].remove(s)
                    moves_used[s] += 1
                    where[s] = potential_school_move[i]
                    cost[0] += distance_matrix[(original[s], where[s])] # calculate initial cost
                    cost_school[original[s]][0] += distance_matrix[(original[s], where[s])]
                    if where[s] in tmpschool_id: # add penalty if student in temporary school
                        cost[0] += P
                        cost_school[original[s]][0] += P
                    break

    damaged_curr = schools_id[ds != 0].copy()

    for d in range(len(order)):# loop through damaged schools and reconstruct
        # update all other values
        membership, moves_used, where, cost_reduction = calculate_potential_tmp(order[d], tmpschool_id, membership, moves_used, original, where, 
                                                   distance_matrix, total_students, classroom, B, M, P)
        cost[d+1] = cost[d] - cost_reduction
        
        for s in range(total_students):
            if original[s] in damaged_schools:
                cost_school[original[s]][d+1] += distance_matrix[(original[s], where[s])] 
                if where[s] in tmpschool_id:
                    cost_school[original[s]][d+1] += P
        
        damaged_curr = damaged_curr[damaged_curr != order[d]] # chosen school no longer damaged
        undamaged_schools = np.append(undamaged_schools, order[d]) # chosen school is now reconstructed
        
    return cost, cost_school

def greedy_capacity_np(schools_id, tmpschool_id, ds, demand, cons_time, classroom, distance_matrix, B, M, P):
    '''
    Function to calculate school reconstruction ordering accounting for capacity, 
    maximum number of moves, and temporary schools. Penalty is separated
    
    Input:
    schools_id: array of school IDs in the region
    tmpschool_id: array of temporary school IDs in the region
    ds: array of damage states for all schools in the region (0,1,2,3)
    demand: dict with keys school ID and values number of students enrolled for each school (includes temp school)
    cons_time: dict of reconstruction time with keys school ID and values the construction time in days
    classroom: dict with keys school ID and values the number of classroom in the corresponding school (includes temp school)
    distance_matrix: a dict with keys school ID pairs, and value the distance between the two schools (includes temp school)
    B: maximum student to classroom ratio
    M: maximum number of moves students can perform
    P: penalty for being in the temporary school
    
    Output:
    order: array of school ID ordered based on the greedy algorithm
    cost: array of costs (weighted-demand)
    '''

    damaged_schools = schools_id[ds != 0]
    num_damaged = len(damaged_schools)
    undamaged_schools = schools_id[ds == 0]
    functional_schools = np.append(undamaged_schools, tmpschool_id) # undamaged + temporary schools

    # total number of students in region
    total_students = sum(demand.values())

    # initial capacity with damaged schools
    initial_capacity = sum([value for key,value in classroom.items() if key in functional_schools])*B

    # initial check if problem is feasible
    if total_students > initial_capacity:
        raise ValueError('Not enough capacity!')

    # Initialize arrays    
    original = np.repeat(schools_id[0], demand[schools_id[0]]) # original schools student went
    for i in range(1, len(schools_id)):
        original = np.append(original, np.repeat(schools_id[i], demand[schools_id[i]])) 
    moves_used = np.zeros(total_students) # number of moves used by students
    where = original.copy()

    # add students to original school
    membership = {school: [] for school in schools_id} 
    for s in range(total_students):
        membership[original[s]].append(s)
    for i in tmpschool_id: # initialize temporary school membership
        membership[i] = []

    # allocate students of damaged schools to nearest functional school, making sure it doesn't exceed capacity
    cost = np.zeros(num_damaged+1)
    cost_np = np.zeros(num_damaged+1)
    for s in range(total_students):
        if original[s] in damaged_schools: # only move students in damaged schools
            potential_school_move = get_nearest_school_order(original[s], distance_matrix, functional_schools)
            for i in range(len(potential_school_move)):
                if len(membership[potential_school_move[i]]) < classroom[potential_school_move[i]]*B: # there is still enough space
                    membership[potential_school_move[i]].append(s)
                    membership[original[s]].remove(s)
                    moves_used[s] += 1
                    where[s] = potential_school_move[i]
                    cost[0] += distance_matrix[(original[s], where[s])] # calculate initial cost
                    cost_np[0] += distance_matrix[(original[s], where[s])]
                    if where[s] in tmpschool_id: # add penalty if student in temporary school
                        cost[0] += P
                    break

    order = []
    last_temp = {}
    damaged_curr = schools_id[ds != 0].copy()
    cost_reduction = np.zeros(num_damaged)
    
    for d in range(num_damaged):# loop through damaged schools and reconstruct
    #     print('School {} out of {}'.format(d, num_damaged))
        potential = np.zeros(len(damaged_curr))

        # calculate gittins index for each damaged school
        index = 0
        for i in damaged_curr:
            _,_,_,potential[index] = calculate_potential_tmp(i, tmpschool_id, membership, moves_used, original, where, 
                                                   distance_matrix, total_students, classroom, B, M, P)
            index += 1

        # choose school with maximum potential
        chosen_school_index = np.argmax(potential/[cons_time[i] for i in damaged_curr])
        chosen_school = damaged_curr[chosen_school_index]
        order.append(chosen_school)

        # update all other values
        membership, moves_used, where, cost_reduction[d] = calculate_potential_tmp(chosen_school, tmpschool_id, membership, moves_used, original,
                                                                             where, distance_matrix, total_students, classroom, B, M, P)
        cost[d+1] = cost[d] - cost_reduction[d]
        
        for s in range(total_students):
            if original[s] in damaged_schools:
                cost_np[d+1] += distance_matrix[(original[s], where[s])] # calculate cost without penalty
        
        damaged_curr = damaged_curr[damaged_curr != chosen_school] # chosen school no longer damaged
        undamaged_schools = np.append(undamaged_schools, chosen_school) # chosen school is now reconstructed
        
        # keep track of the last time period students are in each temporary school
        for i in tmpschool_id:
            if not membership[i] and not i in last_temp.keys(): # no more students in the temporary school
                last_temp[i] = sum([cons_time[j] for j in order])
        
    return order, cost, cost_np, last_temp, cost_reduction

