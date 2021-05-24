import gurobipy as gp
import numpy as np
from gurobipy import GRB


def lp(JOBS, DEMAND, TIMES, DIST, MACHINES):
	'''
	Original Linear Programming Formulation

	Input:
	JOBS: dict with key damaged SCHOOL ID and value cons_time. Length: len(damaged)
	DEMAND: dict with key all SCHOOL ID and value number of students enrolled in each school. Length: len(all schools)
	TIMES: list of time values
	MACHINES: list of contractors ID
	DIST: dict with key pairs of SCHOOL ID and value distance between each pair
	
	Output: 
	dv: dict of decision variables from optimization
	m: gurobi model
	'''

	m = gp.Model('lp')

	# decision variables
	X = JOBS.keys()
	U = DEMAND.keys()
	PAIRS = [(j,k) for j in X for k in X if j < k]
	start = m.addVars(X, name = 's')
	z = m.addVars(X, MACHINES, vtype = GRB.BINARY, name = 'z')
	y = m.addVars(X, TIMES, vtype = GRB.BINARY, name = 'y' )
	r = m.addVars(U, TIMES, vtype = GRB.BINARY, name = 'r' )
	c = m.addVars(X, TIMES, name = 'c' )
	x = m.addVars(PAIRS, vtype = GRB.BINARY, name = 'x' )
	n = m.addVars(X, U, TIMES, vtype = GRB.BINARY, name = 'n' )
	BigM = sum([JOBS[j]for j in X])

	# Constraints
	f1 = m.addConstrs(((sum(y[j,t] for t in list(range(1, len(TIMES)))) == 1) for j in X), name = 'cons1')
	f2 = m.addConstrs((r[j,t] == sum(y[j,i] for i in list(range(1,t+1))) for j in X for t in TIMES[1:]), name = 'cons2')
	f3 = m.addConstrs(((r[s0,t] == 1) for s0 in S0 for t in TIMES), name = 'cons3')
	f4 = m.addConstrs(((start[j] + JOBS[j] == sum(y[j,t]*t for t in TIMES[1:])) for j in X), name = 'cons4')

	d1 = m.addConstrs(((start[j] + JOBS[j] <= start[k] + BigM*(x[j,k] + (1-z[j,mach]) + (1-z[k,mach]))) 
	        for (j,k) in PAIRS for mach in MACHINES), name = 'm_cons1')
	d2 = m.addConstrs(((start[k] + JOBS[k] <= start[j] + BigM*((1-x[j,k]) + (1-z[j,mach]) + (1-z[k,mach]))) 
	         for (j,k) in PAIRS for mach in MACHINES), name = 'm_cons2')

	j1 =  m.addConstrs(((sum(z[j,mach] for mach in MACHINES) == 1) for j in X), name = 'job1')

	c1 = m.addConstrs(((c[j,t] == sum(n[j,k,t]* DEMAND[j] * DIST[j,k] for k in U)) for j in X for t in TIMES), name = 'cost1')
	c2 = m.addConstrs(((sum(n[j,k,t] for k in U) == 1) for j in X for t in TIMES), name = 'cost2')
	c3 = m.addConstrs(((n[j,k,t] <= r[k,t]) for j in X for k in U for t in TIMES), name = 'cost3')

	# Objective:
	m.setObjective(sum(sum(c[j,t] for t in TIMES) for j in X), GRB.MINIMIZE)

	m.optimize()

	# Decision variables results
	dv = {}
	var_name = ['start', 'z', 'y', 'r', 'c', 'x', 'n']
	model_vars = [start, z, y, r, c, x, n]
	for i in range(len(model_vars)):
		results = {}
		for j,val in model_vars[i].items():
			results[j] =  val.X
		dv[var_name[i]] = results

	return m, dv

def relaxed_lp(X, S0, MACHINES, TIMES, B, cons_time, demand, classroom, distance_matrix, print_output = True, MOVES = 10):
	'''
	Relaxed Linear Programming Formulation with capacity constraints

	Input:
	U: array of all schools ID in region
	X: array of damaged schools ID
	S0: array of functional schools ID
	MACHINES: list of contractors ID
	TIMES: list of time values
	B: maximum student-to-classroom ratio
	moves: maximum number of moves
	cons_time: dict with key SCHOOL ID, and value cons_time. length(U)
	demand: dict with key SCHOOL ID and value number of students enrolled in each school. length(U)
	classroom: dict with key SCHOOL ID and value number of classrooms in each school. length(U)
	distance_matrix: dict with key pairs of SCHOOL ID and value distance between each pair

	Output:
	dv: dict of decision variables from optimization
	m: gurobi model
	'''

	U = np.concatenate([X,S0]) # all schools

	m = gp.Model('relaxed_lp')

	# Decision variables
	y = m.addVars(MACHINES, X, TIMES, lb = 0, name = 'y' )
	r = m.addVars(U, TIMES, name = 'r' )
	c = m.addVars(X, TIMES, name = 'c' )
	n = m.addVars(X, U, TIMES, lb = 0, name = 'n' )
	u = m.addVars(X, U, TIMES, name = 'u' )

	# Constraints
	f1 = m.addConstrs(((sum(sum(y[m,j,t] for t in list(range(1, len(TIMES)))) for m in MACHINES) == 1) for j in X), name = 'cons1')
	f2 = m.addConstrs(((sum(cons_time[j]*y[m,j,t] for j in X) <= 1) for m in MACHINES for t in TIMES[1:]), name = 'cons2')
	f3 = m.addConstrs((r[j,t] == sum(sum(y[m,j,i] for i in list(range(1,t+1))) for m in MACHINES) for j in X for t in TIMES[1:]), name = 'cons3')
	f4 = m.addConstrs(((r[s0,t] == 1) for s0 in S0 for t in TIMES), name = 'cons4')
	f5 = m.addConstrs(((r[j,0] == 0) for j in X), name = 'cons5')

	c1 = m.addConstrs(((c[j,t] == sum(n[j,k,t]* demand[j] * distance_matrix[j,k] for k in U)) for j in X for t in TIMES), name = 'cost1')
	c2 = m.addConstrs(((sum(n[j,k,t] for k in U) == 1) for j in X for t in TIMES), name = 'cost2')
	c3 = m.addConstrs(((n[j,k,t] <= r[k,t]) for j in X for k in U for t in TIMES), name = 'cost3')
	c4 = m.addConstrs((cons_time[j] *(r[j,t] - r[j,t-1]) <= 1 for j in X for t in TIMES[1:]), name = 'cost4')

	# maximum capacity constraints
	c4 = m.addConstrs(((sum(n[j,k,t]*demand[j] for j in X) + demand[k] <= B * classroom[k] * r[k,t]) for k in S0 for t in TIMES), name = 'capacity1')
	c5 = m.addConstrs(((sum(n[j,k,t]*demand[j] for j in X) <= B * classroom[k] * r[k,t]) for k in X for t in TIMES), name = 'capacity2')

	# maximum number of moves constraint
	m1 = m.addConstrs((((n[j,k,t] - n[j,k,t-1]) *demand[j]<= u[j,k,t]) for j in X for k in U for t in TIMES[1:]), name = 'move1')
	m2 = m.addConstrs((((n[j,k,t-1] - n[j,k,t]) *demand[j]<= u[j,k,t]) for j in X for k in U for t in TIMES[1:]), name = 'move2')
	m3 = m.addConstrs(((sum(sum(u[j,k,t] for t in TIMES[1:]) for k in U) <= MOVES * demand[j]) for j in X), name = 'move3')

	# Objective:
	m.setObjective(sum(sum(c[j,t] for t in TIMES) for j in X), GRB.MINIMIZE)

	if not print_output:
		m.Params.LogToConsole = 0

	m.optimize()

	# Decision variables results
	dv = {}
	var_name = ['y', 'r', 'c', 'n']
	model_vars = [y,r,c,n]
	for i in range(len(model_vars)):
		results = {}
		for j,val in model_vars[i].items():
			results[j] =  val.X
		dv[var_name[i]] = results

	# construction schedule
	SCHEDULE = {}
	for key,val in y.items():
	    if val.X > 0:
	        
	        if not key[1] in SCHEDULE.keys():
	            SCHEDULE[key[1]] = []
	            
	        if val.X == 1:
	            dict_results = {
	                'start': key[-1],
	                'finish': key[-1] + cons_time[key[1]],
	                'machine': key[0]}
	        else:
	            dict_results = {
	                'start': key[-1],
	                'finish': key[-1]+1,
	                'machine': key[0]}
	        SCHEDULE[key[1]].append(dict_results)

	return m, dv, SCHEDULE

def relaxed_lp_temp(X, S0, ST, MACHINES, TIMES, B, P, cons_time, demand, classroom, distance_matrix, print_output = True, MOVES = 10):
	'''
	Relaxed Linear Programming Formulation with temporary schools

	Input:
	U: array of all schools ID in region
	S0: array of functional schools ID
	ST: array of temporary schools ID
	MACHINES: list of contractors ID
	TIMES: list of time values
	B: maximum student-to-classroom ratio
	MOVES: maximum number of moves
	P: penalty of being in temporary schools
	cons_time: dict with key SCHOOL ID, and value cons_time. length: X + S0
	demand: dict with key SCHOOL ID and value number of students enrolled in each school. length: X + S0 + ST
	classroom: dict with key SCHOOL ID and value number of classrooms in each school. length: X + S0 + ST
	distance_matrix: dict with key pairs of SCHOOL ID and value distance between each pair (includes distance to temp schools)

	Output:
	dv: dict of decision variables from optimization
	m: gurobi model
	'''

	U = np.concatenate([X,S0,ST]) # all schools
	SF = np.concatenate([S0, ST]) # functional schools

	m = gp.Model('relaxed_lp')

	# Decision variables
	y = m.addVars(MACHINES, X, TIMES, lb = 0, name = 'y' )
	r = m.addVars(U, TIMES, name = 'r' )
	c = m.addVars(X, TIMES, name = 'c' )
	n = m.addVars(X, U, TIMES, lb = 0, name = 'n' )
	u = m.addVars(X, U, TIMES, name = 'u' )

	# Constraints
	f1 = m.addConstrs(((sum(sum(y[m,j,t] for t in list(range(1, len(TIMES)))) for m in MACHINES) == 1) for j in X), name = 'cons1')
	f2 = m.addConstrs(((sum(cons_time[j]*y[m,j,t] for j in X) <= 1) for m in MACHINES for t in TIMES[1:]), name = 'cons2')
	f3 = m.addConstrs((r[j,t] == sum(sum(y[m,j,i] for i in list(range(1,t+1))) for m in MACHINES) for j in X for t in TIMES[1:]), name = 'cons3')
	f4 = m.addConstrs(((r[sf,t] == 1) for sf in SF for t in TIMES), name = 'cons4')
	f5 = m.addConstrs(((r[j,0] == 0) for j in X), name = 'cons5')

	c1 = m.addConstrs(((c[j,t] == sum(n[j,k,t]* demand[j] * distance_matrix[j,k] for k in U)) for j in X for t in TIMES), name = 'cost1')
	c2 = m.addConstrs(((sum(n[j,k,t] for k in U) == 1) for j in X for t in TIMES), name = 'cost2')
	c3 = m.addConstrs(((n[j,k,t] <= r[k,t]) for j in X for k in U for t in TIMES), name = 'cost3')
	c4 = m.addConstrs((cons_time[j] *(r[j,t] - r[j,t-1]) <= 1 for j in X for t in TIMES[1:]), name = 'cost4')

	# maximum capacity constraints
	c4 = m.addConstrs(((sum(n[j,k,t]*demand[j] for j in X) + demand[k] <= B * classroom[k] * r[k,t]) for k in SF for t in TIMES), name = 'capacity1')
	c5 = m.addConstrs(((sum(n[j,k,t]*demand[j] for j in X) <= B * classroom[k] * r[k,t]) for k in X for t in TIMES), name = 'capacity2')

	# maximum number of moves constraint
	m1 = m.addConstrs((((n[j,k,t] - n[j,k,t-1]) *demand[j]<= u[j,k,t]) for j in X for k in U for t in TIMES[1:]), name = 'move1')
	m2 = m.addConstrs((((n[j,k,t-1] - n[j,k,t]) *demand[j]<= u[j,k,t]) for j in X for k in U for t in TIMES[1:]), name = 'move2')
	m3 = m.addConstrs(((sum(sum(u[j,k,t] for t in TIMES[1:]) for k in U) <= MOVES * demand[j]) for j in X), name = 'move3')

	# Objective:
	# m.setObjective(sum(sum((c[j,t] + sum(P*t * n[j,k,t] for k in ST)) for t in TIMES) for j in X), GRB.MINIMIZE)
	m.setObjective(sum(sum((c[j,t] + sum(P * n[j,k,t] for k in ST)) for t in TIMES) for j in X), GRB.MINIMIZE)

	if not print_output:
		m.Params.LogToConsole = 0

	m.optimize()

	# Decision variables results
	dv = {}
	var_name = ['y', 'r', 'c', 'n']
	model_vars = [y,r,c,n]
	for i in range(len(model_vars)):
		results = {}
		for j,val in model_vars[i].items():
			results[j] =  val.X
		dv[var_name[i]] = results

	# construction schedule
	SCHEDULE = {}
	for key,val in y.items():
	    if val.X > 0:
	        
	        if not key[1] in SCHEDULE.keys():
	            SCHEDULE[key[1]] = []
	            
	        if val.X == 1:
	            dict_results = {
	                'start': key[-1],
	                'finish': key[-1] + cons_time[key[1]],
	                'machine': key[0]}
	        else:
	            dict_results = {
	                'start': key[-1],
	                'finish': key[-1]+1,
	                'machine': key[0]}
	        SCHEDULE[key[1]].append(dict_results)

	return m, dv, SCHEDULE