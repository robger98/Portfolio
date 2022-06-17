import numpy as np
import copy
import queue
import json
import argparse
import os

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-f','--forward_checking', required=False, action='store_true', help=
    'If added it enables forward checking')
parser.add_argument('-a','--ac3', required=False, action='store_true', help=
    'If added it enables AC3')
parser.add_argument('-r','--random', required=False, action='store_true', help=
    'If added it enables random variable ordering')
parser.add_argument('-m','--mrv', required=False, action='store_true', help=
    'If added it enables MRV variable ordering')
parser.add_argument('--time_test', required=False, action='store_true', help=
    'If added the program only runs the non-algorithmic code. Used for '+
    'testing performance.')
parser.add_argument('-d','--degree', required=False, action='store_true', help=
    'If added it enables MRV with degree variable ordering')
parser.add_argument('-s','--sudoku', required=False, action='store_true', help=
    'Enable the sudoku csp helpers')
parser.add_argument('-g','--graph', required=False, action='store_true', help=
    'Enable the planar graph helpers')
args = parser.parse_args()

# set arg variables
fc = args.forward_checking
ac = args.ac3
r = args.random
m = args.mrv
d = args.degree
t = args.time_test
s = args.sudoku
g = args.graph

# check for adequate args
if(not t):
    if (r and m or r and d or m and d):
        print('Only one variable ordering method can be chosen!')
        exit()
    if (not (r or (m or d))):
        print('One variable ordering method must be chosen!')
        exit()

if(s and g):
    print('Only one problem type can be chosen!')
    exit()

if(not (s or g)):
    print('One problem type must be chosen!')
    exit()

# The not neq lambda function
neq = (lambda x,y: x != y)

# Constraint class: holds variables in specific order
class Constraint():
    def __init__(self, x, y, constraint_func):
        self.x = x
        self.y = y
        self.c = constraint_func

    def copy(self):
        return Constraint(self.x, self.y, self.c)

    # Take in a dictionary of variables and their values. ex. {x:1. y:2}
    # this allows us to not care about the order of the variables in the lambda 
    # function
    def satisfied(self, var_values_dict):
        output = self.c(var_values_dict[self.x], var_values_dict[self.y])
        return output
    
    # Returns if the constraint contains the given variable
    def contains(self, variable):
        if (self.x == variable):
            return 0
        elif (self.y == variable):
            return 1
        
        return -1

# This class contains the CSP data and different inference and ordering 
# hueristics
class CSP():
    def __init__(self, variables, constraints, domains):
        self.v = variables      # variable names
        self.c = constraints    # constraints between the variables
        self.d = domains        # Dictionary of the variables and their possible
                                    # values
        self.a = {}             # The starting assignment
    
    # Make a deep copy of this CSP. Useful for backtracking
    def copy(self):
        csp = CSP(copy.deepcopy(self.v), copy.deepcopy(self.c), copy.deepcopy(self.d))
        csp.a.update(copy.deepcopy(self.a))
        return csp

    # Adds an assignment if the value is in the variables domain
    def assign(self, variable, value):
        if value in self.d[variable]:
            self.a[variable] = value

    # Checks if all assigned variables satifies their fully assigned constraints
    def is_consistent(self):
        for var in self.a.keys():
            cons = self.get_constraints_with(var) # get the constraints with var
            for c in cons:
                # get the other variable in the constraint
                other_var = None
                if c[1] == 0:
                    other_var = c[0].y
                else:
                    other_var = c[0].x
                
                # check if the other variable is assigned
                if self.is_assigned(other_var):
                    # If the constraint is not satisfied return false
                    if not c[0].satisfied({var: self.a[var], other_var: self.a[other_var]}):
                        return False
        return True

    # returns the list of variables constrained with the given variable (no duplicates)
    def get_neighbors(self, variable):
        rel_cons = self.get_constraints_with(variable)
        neighbors = []
        for con in rel_cons:
            if con[1] == 0:
                if (con[0].y not in neighbors):
                    neighbors.append(con[0].y)
            if con[1] == 1:
                if (con[0].x not in neighbors):
                    neighbors.append(con[0].x)
        return neighbors

    # Returns the list constraints that contain the variable
    # Output is a list of 2 element lists. 
    # Each two element list follows the format [<constraint>, <constraint.contains(variable)>] 
    # where the second element signifies if the variable is x or y in the constraint.
    def get_constraints_with(self, variable):
        relevant_constraints = []
        for cons in self.c:
            x = cons.contains(variable)
            if x != -1:
                relevant_constraints.append([cons, x])
        
        return relevant_constraints
    
    # Returns a list of only constraints with both variables.
    # Output is a list of 2 element lists. 
    # Each two element list follows the format [<constraint>, <constraint.contains(var1)>] 
    # where the second element signifies if var1 is x or y in the constraint.
    def get_constraints_with_both(self, var1, var2):
        first_order = []
        relevant_constraints = []
        first_order = self.get_constraints_with(var1)
        for cons in first_order:
            if cons[0].contains(var2) != -1:
                relevant_constraints.append([cons[0], cons[1]])
        
        return relevant_constraints
    # Returns if a variable is assigned
    def is_assigned(self, variable):
        return (variable in self.a.keys())

    # computes forward checking
    def forward_check(self, variable):
        # get the constraints
        cons_with_order = self.get_constraints_with(variable)
        for c in cons_with_order:
            other_var = None
            if c[1] == 0:
                other_var = c[0].y
            else:
                other_var = c[0].x
            # Check if the other constrained variable is assigned
            if not self.is_assigned(other_var):
                # if assigned loop through the variables domain values
                for val in list(self.d[other_var]):
                    # if the value does not satisfy the constraint remove it
                    if not c[0].satisfied({variable: self.a[variable], other_var: val}):
                        self.d[other_var].remove(val)
        return {}
    
    # Revises the domain of var1 given the constraints with var2
    def _revise(self, var1, var2):
        revised = False
        cons = self.get_constraints_with_both(var1, var2)
        # check if constraint exists between variables
        if len(cons) == 0:
            # if not return false
            return False
        # for each variable in var1's domain check if its constraint with var2
        # is satisfiable
        for x in list(self.d[var1]):
            satisfiable = False
            for y in self.d[var2]:
                for constraint in cons:
                    if constraint[0].satisfied({var1: x, var2: y}):
                        satisfiable = True
            # if not satisfiable remove the value from var1's domain
            if not satisfiable:
                self.d[var1].remove(x)
                revised  = True
        # return whether the domain was revised
        return revised
    # Performs the ac3 algorithm on this CSP object
    def ac3(self):
        # Create the queue
        q = []
        # Add each constraints variables in the queue
        for cons in self.c:
            q.append((cons.x, cons.y))
            q.append((cons.y, cons.x))
        # Loop through the queue taking the tuple of variable names and revise them
        while len(q) > 0:
            (xi, xj) = q[0]
            #print ("AC-3 checking " + str(xi) + "---" + str(xj))
            #input()
            q.remove(q[0])
            # if the domain of xi is revised but not empty
            if self._revise(xi, xj):
                if len(self.d[xi]) == 0:
                    # if empty the problem is not solvable and return false
                    return False
                # get the neighbors of xi and append them into the queue with xi
                else:
                    for xk in self.get_neighbors(xi):
                        if xk != xj:
                            q.append((xk, xi))
        return True
    # get all unassigned variables
    def get_unassigned(self):
        unassigned = []
        for var in self.v:
            if not self.is_assigned(var):
                unassigned.append(var)\
        
        return unassigned

    #Return a random unassigned variable
    def random_vo(self):
        unassigned = self.get_unassigned()
        return unassigned[np.random.randint(0, len(unassigned))]

    # Return the unassigned variable with the least values in its domain
    def mrv_vo(self):
        unassigned = self.get_unassigned()
        min_remaining = None
        min_var = None
        # loop through the unassigned variables and record the min domain values
        for var in unassigned:
            if min_remaining == None or min_remaining > len(self.d[var]):
                min_remaining = len(self.d[var])
                min_var = var
        return min_var

    # Return how many unassigned constraints the variable is involved with
    def degree(self, var):
        unassigned_constraints_counter = 0
        # find all the constraints with var
        constraints = self.get_constraints_with(var)
        # if either variable is unassigned add 1 to the counter
        for cons in constraints:
            if (not self.is_assigned(cons[0].x) and not self.is_assigned(cons[0].y)):
                unassigned_constraints_counter += 1
        return unassigned_constraints_counter
    # Return the minimum remaining value with highest degree unassigned variable
    def mrv_vo_with_degree(self):
        unassigned = self.get_unassigned()
        min_remaining = None
        min_vars = []
        for var in unassigned:
            # set if its the first var, append if it matches, and overwrite if 
            # it beat the previous min
            if (min_remaining == None):
                min_remaining = len(self.d[var])
                min_vars.append(var)
            elif min_remaining == len(self.d[var]):
                min_vars.append(var)
            elif min_remaining > len(self.d[var]):
                min_remaining = len(self.d[var])
                min_vars = [var]
        
        # Loop through and track the highest degree variable from the MRV variables
        max_deg = self.degree(min_vars[0])
        max_deg_var = min_vars[0]
        for var in min_vars:
            if max_deg < self.degree(var):
                max_deg = self.degree(var)
                max_deg_var = var

        return max_deg_var

    def print_constraints(self):
        for c in self.c:
            print(str(c.x) + '-----' + str(c.y))

def __main():
    # Set dir for sudoku and gcp
    local_dir = os.path.join("D:", os.sep, "vscode_workspace", "ai-class", "AI_Class_Workspace", "Robert_Geraghty_Project_3")
    sudoku = local_dir + "\\sudoku.json"
    graph = local_dir +"\\gcp.json"
    # init domains
    if(s):
        domains = sudoku_domain_helper(open(sudoku))
        
    else:
        domains = map_domain_helper(open(graph))
    
    # init vars
    x = []
    for v in domains.keys():
        x.append(v)

    #init constraints
    if(s):
        constraints = sudoku_constraint_helper()
    else:
        constraints = map_constraint_helper(open(graph))

    # create csp
    csp = CSP(x, constraints, domains)
    # if csp is sudoku create inital assignments
    if(s):
        csp.a = sudoku_assignment_helper(csp, open(sudoku))
    if(not t):
        print(constraint_satisfaction(csp))

# reads in the json variables and adds the four colors to the domain
def map_domain_helper(json_file):
    domain = {}
    data = json.load(json_file)
    for p in data['points'].keys():
        domain[int(p)] = [1, 2, 3, 4]
    return domain

# Reads in the json edges and creats a not equals constraint between the two vars
def map_constraint_helper(json_file):
    constraints = []
    data = json.load(json_file)
    for e in data['edges']:
        c = Constraint(e[0], e[1], neq)
        constraints.append(c)
    return constraints

# Creates 81 vars and makes their domain their starting assignment if assigned
# or 1-9 if not assigned
def sudoku_domain_helper(json_file):
    domains = {}
    board = json.load(json_file)
    for row in range(0, 9):
        for col in range(0, 9):
            # if assigned limit domain to the starting value
            if board[row][col] != 0:
                domains[row*10+col] = [board[row][col]]
            else:
                domains[row*10+col] = [1,2,3,4,5,6,7,8,9]
    return domains

# Make row wise, column wise, and square wise constraints for each variable. 
# may be slightly overkill but it is complete
def sudoku_constraint_helper():
    constraints = []
    
    #handle row wise neq
    for row in range(0, 9):
        for col1 in range (0, 9):
            for col in range(col1+1 ,9):
                constraints.append(Constraint((row*10+col1), (row*10+col), neq))
            
    #handle col wise neq
    for col in range(0,9):
        for row1 in range(0, 9):
            for row in range(row1+1, 9):
                constraints.append(Constraint((row1*10+col), (row*10+col), neq))
    
    #handle block wise neq
    for block_row in [0,3,6]:
        for block_col in [0,3,6]:
            for row1 in range(0,3):
                for col1 in range(0,3):
                    for row in range(0,3):
                        for col in range(0,3):
                            if(row != row1 or col != col1):
                                constraints.append(Constraint((block_row+row1)*10+block_col+col1, (block_row+row)*10+block_col+col, neq))
    return constraints

# assigns the starting values to their variables in the CSP
def sudoku_assignment_helper(csp, json_file):
    board = json.load(json_file)

    for row in range(0,9):
        for col in range(0,9):
            if board[row][col] != 0:
                csp.a[row*10+col] = board[row][col]
    return csp.a

# If the AC-3 option is selected it will run ac3, otherwise it just return 
# the completed assignment from the depth first algorithm
def constraint_satisfaction(csp):
    if(ac):
        csp.ac3()
    return depth_first_with_backtracking(csp)

# performs depth first backtracking on the csp
def depth_first_with_backtracking(csp):
    # make a back up of the CSP for backtracking
    backup_csp = csp.copy()
    # Check if the assignment is complete
    if complete_check(csp.a, csp.v):
        print ('complete!')
        return csp.a
    #Choose the variable ordering method
    if(r):
        variable = csp.random_vo()
    elif(m):
        variable = csp.mrv_vo()
    elif(d):
        variable = csp.mrv_vo_with_degree()

    #For each value in the variables domain
    for value in order_domain_values(variable, csp.a, csp):
        # Deep copy from the backup to the working csp
        csp = backup_csp.copy()
        # assign the variable
        csp.assign(variable, value)
        # check if the assignment is consistent
        if csp.is_consistent():
            # if it is perform inference
            inferences = inference(csp, variable)
            # if the inference algorithm does not return null than update 
            # the current assignment with the inferences and recurse with the 
            # newly updated csp
            if (inferences != None):
                csp.a.update(inferences)
                result = depth_first_with_backtracking(csp)
                if result != None:
                    return result

# if forward checking is selected it will run otherwise it will return an empty
# dictionary
def inference(csp,var):
    if (fc):
        return csp.forward_check(var)
    else:
        return {}

def complete_check(assignment, variables):
    assignment_set = set()
    variable_set = set()

    for var in assignment.keys():
        assignment_set.add(var)
    
    for var in variables:
        variable_set.add(var)

    return assignment_set == variable_set
                
# Only returns the variable's domain
def order_domain_values(var, assignment, csp):
    return csp.d[var]

if __name__ == "__main__":
    __main()

    
            





        
        