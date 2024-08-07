import numpy as np
from itertools import chain, combinations, permutations
from gurobipy import Model, GRB, quicksum

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

def compute_nash_welfare(valuation, allocation):
    """
    Compute the Nash welfare for a given allocation.
    
    Parameters:
        valuation (numpy.ndarray): A 2D array where valuation[i][j] is the value agent i assigns to item j.
        allocation (numpy.ndarray): A matrix representing an allocation of items to agents.
        
    Returns:
        float: The Nash welfare of the allocation.
    """
    utility = np.dot(valuation, allocation.T).diagonal()  # Calculate utility for each agent
    nw = np.prod(utility)  # Calculate Nash welfare as the product of utilities
    return nw

def maximize_nash_welfare(n, m, valuations):
    # Create a new model
    model = Model("NashWelfare")

    # Suppress the output
    model.setParam('OutputFlag', 0)

    # Create variables
    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
    U = model.addVars(n, vtype=GRB.CONTINUOUS, name="U")
    logU = model.addVars(n, vtype=GRB.CONTINUOUS, name="logU")
    z = model.addVar(vtype=GRB.CONTINUOUS, name="z")

    # Set objective to maximize z (logarithm of Nash welfare)
    model.setObjective(z, GRB.MAXIMIZE)

    # Add constraints
    # Each item is allocated to exactly one agent
    model.addConstrs((quicksum(x[i, j] for i in range(n)) == 1 for j in range(m)), "item_allocation")

    # Define the utility for each agent
    model.addConstrs((U[i] == quicksum(valuations[i][j] * x[i, j] for j in range(m)) for i in range(n)), "agent_utility")

    # Linearize the logarithm of utilities
    model.addConstrs((logU[i] <= U[i] for i in range(n)), "log_utility")

    # Link the logarithm of the Nash welfare to individual utilities
    model.addConstrs((z <= logU[i] for i in range(n)), "nash_link")

    # Optimize the model
    model.optimize()

    # Extract the optimal allocation
    allocation = np.zeros((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            if x[i, j].X > 0.5:
                allocation[i][j] = 1

    return allocation
