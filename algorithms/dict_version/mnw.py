from itertools import combinations, chain, permutations, product
import numpy as np
import numpy as np
from itertools import permutations, combinations
from scipy.special import comb

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def generate_distributions_optimized(n_items, n_agents):
        """
        Generate all possible distributions of n_items to n_agents in an optimized manner.
        """
        if n_agents == 1:
            yield (n_items,)
        else:
            for first_agent_items in range(n_items + 1):
                for rest in generate_distributions_optimized(n_items - first_agent_items, n_agents - 1):
                    yield (first_agent_items,) + rest

# Correcting the distribute_items_according_to_distribution_optimized method
# to make it more efficient by using sets for removal and adding sorting only when yielding results.
def distribute_items_according_to_distribution_optimized(items, distribution):
    """
    Generate all unique sets of items for each distribution more efficiently by using sets for faster removal.
    """
    if len(distribution) == 1:
        yield [tuple(items)]
    else:
        first_agent_items = distribution[0]
        items_set = set(items)
        for items_for_first_agent in combinations(items, first_agent_items):
            remaining_items_set = items_set - set(items_for_first_agent)
            for subsequent_allocation in distribute_items_according_to_distribution_optimized(sorted(remaining_items_set), distribution[1:]):
                yield [tuple(sorted(items_for_first_agent))] + subsequent_allocation



# This approach uses sets for removal and adds sorting only when yielding results to ensure consistency.
# It should significantly reduce the overhead of the original list removal process.


def all_allocations(n_items, n_agents):
    items = list(range(n_items))
    agents = list(range(n_agents))
    # Generate all possible ways to distribute the number of items to agents
    distributions = generate_distributions_optimized(n_items, n_agents)

    for distribution in distributions:
        # For each distribution, generate all unique sets of items for the agents
        for unique_set in distribute_items_according_to_distribution_optimized(items, distribution):
            allocation = {agent: bundle for agent, bundle in zip(agents, unique_set)}
            yield allocation

def compute_nash_welfare(valuations, allocation):
    """
    Compute the Nash Welfare for a given allocation.
    """
    nw = 0
    agent_valuation = [0]*len(valuations)
    for agent, items in allocation.items():
        agent_valuation[agent] = sum(valuations[agent][item] for item in items)
    if any(val == 0 for val in agent_valuation):
        return -np.inf
    else:
        nw = np.sum(agent_valuation)
    return nw

def maximize_nash_welfare(n_agents, m_items, valuations):
    """
    Find the allocation that maximizes the Nash Welfare.
    """
    agent = list(range(n_agents))
    items = list(range(m_items))
    max_nw = 0
    best_allocations = []

    for allocation in all_allocations(m_items, n_agents):
        nw = compute_nash_welfare(valuations, allocation)
        if nw > max_nw:
            max_nw = nw

            best_allocations = [allocation]
        elif nw == max_nw:
            best_allocations.append(allocation)

    return best_allocations

