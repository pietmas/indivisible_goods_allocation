import numpy as np
from itertools import chain, combinations, permutations
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

def maximize_nash_welfare(n_agents, m_items, valuation):
    """
    Find the allocation that maximizes the Nash welfare.
    
    Parameters:
        n_agents (int): The number of agents.
        m_items (int): The number of items.
        valuation (numpy.ndarray): A 2D array of valuations.
    
    Returns:
        list: A list of matrices representing the best allocations.
    """
    max_nw = 0
    best_allocations = []

    # Iterate through all possible allocations
    for allocation in all_allocations(m_items, n_agents):
        nw = compute_nash_welfare(valuation, allocation)
        if nw > max_nw:
            max_nw = nw
            best_allocations = [allocation]  # Update best allocation if a better one is found
        elif nw == max_nw:
            best_allocations.append(allocation)  # Add to best allocations if it's equally good

    return best_allocations
