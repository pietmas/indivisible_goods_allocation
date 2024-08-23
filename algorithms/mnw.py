import numpy as np
from itertools import chain, combinations, permutations
from scipy.optimize import linear_sum_assignment
from gurobipy import Model, GRB, quicksum
import math

"""
These functions are used to compute the maximum Nash welfare allocation for a given valuation matrix.
The maximum Nash welfare allocation is the allocation that maximizes the product of utilities for all agents.

The maximum Nash welfare allocation is EF1 and PO, as was proven in
"The Unreasonable Fairness of Maximum Nash Welfare"
by Ioannis Caragiannis, David Kurokawa, Hervé Moulin, Ariel D. Procaccia, Nisarg Shah, Junxing Wang

Link: https://dl.acm.org/doi/10.1145/3355902

"""
def powerset(iterable):
    """
    Generate all possible subsets (the power set) of the given iterable.
    
    Parameters:
        iterable (iterable): The input iterable.
        
    Returns:
        chain: An iterable of tuples representing the power set.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def generate_distributions(n_items, n_agents):
    """
    Generate all possible distributions of n_items to n_agents.
    
    Parameters:
        n_items (int): The number of items.
        n_agents (int): The number of agents.
        
    Yields:
        tuple: A tuple representing a distribution of items among agents.
    """
    if n_agents == 1:
        yield (n_items,)
    else:
        for first_agent_items in range(n_items + 1):
            for rest in generate_distributions(n_items - first_agent_items, n_agents - 1):
                yield (first_agent_items,) + rest

def distribute_items_according_to_distribution(items, distribution):
    """
    Generate all unique sets of items for each distribution.
    
    Parameters:
        items (list): The list of items.
        distribution (tuple): A tuple representing the distribution of items among agents.
        
    Yields:
        list: A list of tuples representing the allocation of items to agents.
    """
    if len(distribution) == 1:
        yield [tuple(items)]
    else:
        first_agent_items = distribution[0]
        items_set = set(items)
        for items_for_first_agent in combinations(items, first_agent_items):
            remaining_items_set = items_set - set(items_for_first_agent)
            for subsequent_allocation in distribute_items_according_to_distribution(
                    sorted(remaining_items_set), distribution[1:]):
                yield [tuple(sorted(items_for_first_agent))] + subsequent_allocation

def all_allocations(n_items, n_agents):
    """
    Generate all possible allocations of items to agents.
    
    Parameters:
        n_items (int): The number of items.
        n_agents (int): The number of agents.
        
    Yields:
        numpy.ndarray: A matrix representing an allocation of items to agents.
    """
    distributions = generate_distributions(n_items, n_agents)
    items = list(range(n_items))
    for distribution in distributions:
        for unique_set in distribute_items_according_to_distribution(items, distribution):
            allocation_matrix = [[0]*n_items for _ in range(n_agents)]
            for agent_index, bundle in enumerate(unique_set):
                for item in bundle:
                    allocation_matrix[agent_index][item] = 1
            allocation_matrix = np.array(allocation_matrix)
            yield allocation_matrix

def compute_nash_welfare(valuations, allocation):
        """
        Compute the Nash Welfare for a given allocation.

        Args:
            allocation (numpy.ndarray): The allocation matrix where each row represents an agent 
                                        and each column represents an item.

        Returns:
            float: The Nash Welfare of the given allocation.
        """
        num_agents, num_items = allocation.shape
        # Calculate the utility for each agent
        utility = np.dot(valuations, allocation.T).diagonal()
        log_utility = np.zeros(len(valuations))
        for i in range(num_agents):
            if utility[i] == 0:
                log_utility[i] = - float("inf")
            else:
                log_utility[i] = np.log(utility[i]) 
        nw = np.sum(log_utility)  # Calculate Nash welfare as the product of utilities
        return nw
        
    
def maximize_nash_welfare_bruteforce(n, m, valuation):
    """
    Compute the maximum Nash welfare allocation for a given valuation matrix using brute force.
    
    Parameters:
        n (int): Number of agents.
        m (int): Number of items.
        valuation (numpy.ndarray): A 2D array where valuation[i][j] is the value agent i assigns to item j.
        
    Returns:
        list of numpy.ndarray: A list of matrices representing the maximum Nash welfare allocations.
    """

    # Initialize variables to track the maximum Nash welfare and the corresponding allocations.
    max_nash_welfare = -1
    max_nash_welfare_allocation = []

    # If there are fewer items than agents, use a different strategy.
    if m < n:
        # Extract the relevant submatrix of valuations for the first 'm' items and 'n' agents.
        valuation_matrix = valuation[np.ix_(n, m)]
        allocation = np.zeros((n, m), dtype=int)  # Initialize an allocation matrix with zeros.
        
        # Use a function to maximize product matching with dummy agents/items to handle the allocation.
        agents, items, max_prod = maximize_product_matching_with_dummy(valuation_matrix)
        
        # Assign items to agents based on the optimal matching.
        for a, i in list(zip(agents, items)):
            allocation[a, i] = 1
        
        # Add the resulting allocation to the list of max Nash welfare allocations.
        max_nash_welfare_allocation.append(allocation)
    
    else:
        # When there are at least as many items as agents, check all possible allocations.
        for allocation in all_allocations(m, n):
            # Compute the Nash welfare for the current allocation.
            nw = compute_nash_welfare(valuation, allocation)
            
            # If the current Nash welfare is greater than the recorded maximum, update the max and reset the list.
            if nw > max_nash_welfare:
                max_nash_welfare = nw
                max_nash_welfare_allocation = [allocation]
            
            # If the current Nash welfare equals the recorded maximum, add the allocation to the list.
            if nw == max_nash_welfare:
                max_nash_welfare_allocation.append(allocation)
    
    # Return the list of allocations that achieve the maximum Nash welfare.
    return max_nash_welfare_allocation

def maximize_product_matching_with_dummy(weights):
        n, m = weights.shape

        # Step 1: Transform weights to logarithms
        log_weights = np.log(weights)
        
        # Step 2: Add dummy items with log(1) = 0 weight
        if n > m:
            dummy_weights = np.zeros((n, n - m))
            log_weights = np.hstack((log_weights, dummy_weights))
        
        # Step 3: Use the Hungarian algorithm to find the maximum weight matching
        row_ind, col_ind = linear_sum_assignment(-log_weights)  # scipy's function finds the min cost assignment, so use -log_weights
        
        # Step 4: Filter out the dummy matches
        valid_matches = col_ind < m
        matched_agents = row_ind[valid_matches]
        matched_items = col_ind[valid_matches]
        
        # Step 5: Calculate the maximum product from the matching
        max_product = np.prod(weights[matched_agents, matched_items])
        
        return matched_agents, matched_items, max_product



def maximize_nash_welfare_milp(n, m, valuation_matrix):
    """
    This function maximizes the Nash social welfare given a set of agents and items.
    
    Parameters:
    - n: The number of agents.
    - m: The number of items.
    - valuation_matrix: A matrix of size n x m where valuation_matrix[i, j] represents
                        the valuation of agent i for item j.
    
    The function returns an allocation matrix of size n x m where allocation[i, j] is 1
    if item j is allocated to agent i, and 0 otherwise. The objective is to maximize 
    the sum of the logarithms of the utilities (Nash social welfare) while ensuring 
    each item is allocated to exactly one agent.
    
    The function employs a piecewise linear approximation to the log function to handle 
    the non-linearity while ensuring that the optimization problem remains tractable 
    as a mixed-integer linear program (MILP). The approximation uses segments connecting 
    points (k, log(k)) and (k+1, log(k+1)) for k in [1, 999] to tightly bound the log 
    function at each integer point.

    Algorithm proposed by Caragiannis et al. (2019) in "The Unreasonable Fairness of Maximum Nash Welfare".
    Link: https://dl.acm.org/doi/10.1145/3355902
    """
    
    # List of agents and items
    agents = list(range(n))
    items = list(range(m))

    # Initialize the Gurobi model
    model = Model()
    model.setParam('OutputFlag', 0)  # Mute Gurobi output

    # Binary decision variables: x[i, j] indicates whether agent i receives item j
    x = model.addVars(agents, items, vtype=GRB.BINARY)

    # Constraint: Each item must be assigned to exactly one agent
    model.addConstrs(quicksum(x[i,j] for i in agents) == 1 for j in items)

    # Continuous variables representing the log utility for each agent
    W = model.addVars(agents)  # log utility

    # Iterate over each agent
    for i in agents:
        # Utility for agent i: sum of valuations for the items assigned to them
        u = quicksum(valuation_matrix[i,j] * x[i,j] for j in items)

        # Apply the piecewise linear approximation of the log function
        # We use segments connecting points (k, log(k)) and (k+1, log(k+1)) for k in [1, 999]
        # This ensures that W[i] is an upper bound on the log utility at every integer point
        for k in range(1, 200):
            model.addConstr(W[i] <= math.log(k) + (math.log(k+1) - math.log(k)) * (u - k))
    
    # Objective: Maximize the sum of log utilities (i.e., Nash welfare)
    model.setObjective(quicksum(W[i] for i in agents), GRB.MAXIMIZE)
    model.optimize()

    # Extract the optimal allocation from the binary variables x[i, j]
    allocation = np.zeros((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            if x[i, j].X > 0.5:
                allocation[i][j] = 1

    return allocation
