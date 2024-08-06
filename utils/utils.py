import random
import string
import numpy as np

def randint_valuation(n, m, value_range=[0, 10]):
    """
    Generate n agents, m items, and a random valuation matrix where each row corresponds to an agent and each column to an item.

    Parameters:
    n (int): Number of agents.
    m (int): Number of items.

    Returns:
    np.array: A matrix of shape (n, m) containing random valuations for each item for each agent.
    """
    # Generate the random valuation matrix
    valuation_matrix = np.random.randint(value_range[0], value_range[1], size=(n, m))
    
    return valuation_matrix

def generate_bivalue_valuations(n, m):
    """
    Generate random valuations for each agent-item pair where valuation is either 0 or 1.
    
    :param agents: list of agent identifiers.
    :param items: list of item identifiers.
    :return: dict of dicts, utilities[agent][item] = valuation (0 or 1).
    """
    valuations = {}
    # Generate agent names (A1, A2, ..., An)
    agents = ['A' + str(i) for i in range(1, n + 1)]
    
    # Generate item names (Item1, Item2, ..., Itemm)
    items = ['G' + str(i) for i in range(1, m + 1)]
    for agent in agents:
        valuations[agent] = {item: random.randint(0, 1) for item in items}
    return agents, items, valuations

def print_allocation_values(allocation, valuations):
    for agent, allocated_items in allocation.items():
        total_value = sum(valuations[agent][item] for item in allocated_items)
        print(f"{agent}'s allocation value: {total_value}")
    return 