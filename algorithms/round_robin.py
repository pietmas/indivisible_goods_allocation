import numpy as np
from itertools import chain, combinations, permutations
from utils.visualization import bipartite_graph_visualization

def round_robin_allocation(n_agents, n_items, valuations):
    """
    Distribute n_items to n_agents in a round-robin fashion considering their valuations.

    Parameters:
    n_agents (int): Number of agents among whom the items are to be allocated.
    n_items (int): Number of items to be allocated.
    valuations (dict): A nested dictionary where valuations[agent][item] is the valuation
                       of the item by the agent.

    Returns:
    dict: A dictionary mapping each agent to their allocated items.
    """
    agents = list(range(n_agents))
    items = list(range(n_items))

    allocation = np.zeros((n_agents, n_items), dtype=int)  # Initialize the allocation matrix
    unallocated_items = items[:]  # List of items to be allocated
    turn_order = np.random.permutation(n_agents)  # Randomly determine the turn order of agents

    # Allocate items until all items are distributed
    while unallocated_items:
        for agent in turn_order:
            if not unallocated_items:
                break
            # Find the most preferred item for the current agent
            preferred_item = max(unallocated_items, key=lambda item: valuations[agent][item])
            # Allocate the item to the agent
            allocation[agent][preferred_item] = 1
            unallocated_items.remove(preferred_item)  # Remove the allocated item from the list
        
    return allocation

def all_possible_round_robin_allocations(n_agents, n_items, valuation_matrix):
    """
    Generate all possible allocations of items to agents based on all permutations of agent order,
    considering their valuations of the items, where valuations are provided as a matrix.
    Allocation results are presented as 0,1-matrices.

    Parameters:
    n_agents (int): Number of agents among whom the items are to be allocated.
    n_items (int): Number of items to be allocated.
    valuation_matrix (np.array): A 2D numpy array where valuation_matrix[agent][item] is the valuation
                                 of the item by the agent.

    Returns:
    list of np.array: A list containing 0,1-matrices, each representing the allocation
                      of items to agents for each permutation of agent orders.
    """
    agents = list(range(n_agents))
    items = list(range(n_items))
    all_allocations = []  # List to store all possible allocations
    order = []  # List to store agent orders

    # Generate all permutations of agent orders
    for agent_order in permutations(agents):
        allocation = np.zeros((n_agents, n_items), dtype=int)  # Initialize allocation matrix for the current permutation
        unallocated_items = items[:]  # List of items to be allocated
        
        # Allocate items until all items are distributed
        while unallocated_items:
            for agent in agent_order:
                if not unallocated_items:
                    break
                # Find the most preferred item for the current agent
                preferred_item = max(unallocated_items, key=lambda item: valuation_matrix[agent][item])
                allocation[agent][preferred_item] = 1  # Allocate the item to the agent
                unallocated_items.remove(preferred_item)  # Remove the allocated item from the list

        all_allocations.append(allocation)  # Add the allocation result for this permutation to the list
        order.append(agent_order)  # Add the agent order to the list

    return all_allocations, order
