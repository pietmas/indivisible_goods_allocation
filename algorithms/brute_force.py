from itertools import combinations, chain, permutations, product
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial




class BruteForce:
    """
    Class to compute all the pareto-optimal (PO) and envy-free up to one item (EF1) allocaiton.

    This class provides methods to compute all the pareto-optimal (PO) and envy-free up to one item (EF1) allocations
    in a brute-force manner for a given number of agents and items.

    It generates all possible allocations of items to agents and checks if each allocation is both EF1 and PO.
    """
    def __init__(self, n_agents, n_items, valuation):
        self.nitems = n_items
        self.nagents = n_agents
        self.valuation = valuation
        self.agents = list(range(n_agents))
        self.items = list(range(n_items))
        self.allocation_utilities = {}
        self.pareto_optimal_utilities = []

    def generate_distributions(self, n_items, n_agents):
        """
        Generate all possible distributions of n_items to n_agents in an optimized manner.

        Parameters:
        n_items (int): The total number of items to be distributed.
        n_agents (int): The total number of agents among whom the items will be distributed.

        Yields:
        tuple: A tuple representing one possible distribution of items among the agents.
            Each element in the tuple corresponds to the number of items assigned to an agent.
        """
        
        # Base case: If there's only one agent, all items are allocated to this single agent
        if n_agents == 1:
            yield (n_items,)
        else:
            # Iterate over possible item counts for the first agent
            for first_agent_items in range(n_items + 1):
                # Recursively generate distributions for the remaining items and agents
                for rest in self.generate_distributions(n_items - first_agent_items, n_agents - 1):
                    # Yield the current distribution as a tuple
                    yield (first_agent_items,) + rest

    def distribute_items_according_to_distribution(self, items, distribution):
        """
        Generate all unique sets of items for each distribution more efficiently.

        Parameters:
        items (list): The list of items to be distributed.
        distribution (list): A list where each element represents the number of items each agent should receive.

        Yields:
        list: A list of tuples where each tuple represents a unique set of items distributed according to the distribution.
        """
        
        # Base case: If the distribution has only one element, yield all remaining items as a single set
        if len(distribution) == 1:
            yield [tuple(items)]
        else:
            # Number of items the first agent should receive
            first_agent_items = distribution[0]
            
            # Convert items list to a set for efficient removal
            items_set = set(items)
            
            # Iterate over all possible combinations of items for the first agent
            for items_for_first_agent in combinations(items, first_agent_items):
                
                # Calculate the remaining items after allocating to the first agent
                remaining_items_set = items_set - set(items_for_first_agent)
                
                # Recursively distribute the remaining items to the subsequent agents
                for subsequent_allocation in self.distribute_items_according_to_distribution(
                        sorted(remaining_items_set), distribution[1:]):
                    
                    # Yield the current allocation combined with subsequent allocations
                    yield [tuple(sorted(items_for_first_agent))] + subsequent_allocation


    def all_allocations(self):
        """
        Generate all possible allocations of items to agents.

        Yields:
        np.array: A matrix representation of the allocation where each row corresponds to an agent
                and each column corresponds to an item. A value of 1 indicates that the item is
                allocated to the agent, and 0 otherwise.
        """
        
        n_items, n_agents = self.nitems, self.nagents

        # Generate all possible ways to distribute the number of items to agents
        distributions = self.generate_distributions(n_items, n_agents)

        for distribution in distributions:
            # For each distribution, generate all unique sets of items for the agents
            for unique_set in self.distribute_items_according_to_distribution(self.items, distribution):
                # Initialize an allocation matrix with zeros
                allocation_matrix = np.zeros((n_agents, n_items), dtype=int)
                
                # Fill the allocation matrix based on the unique set of items for each agent
                for agent_index, bundle in enumerate(unique_set):
                    for item in bundle:
                        item_index = self.items.index(item)
                        allocation_matrix[agent_index][item_index] = 1
                
                # Yield the allocation matrix
                yield allocation_matrix

    def is_ef1(self, allocation):
        """
        Check if the given allocation is EF1 (Envy-Free up to one item).

        Parameters:
        allocation (dict): A dictionary mapping each agent to their allocated items.
        valuations (dict): A nested dictionary where valuations[agent][item] is the valuation
                        of the item by the agent.

        Returns:
        bool: True if the allocation is EF1, False otherwise.
        """
        utility_matrix = np.dot(self.valuation, np.array(allocation).T)
        # Iterate through each pair of agents to check the EF1 condition
        for agent_i in self.agents:
            for agent_j in self.agents:
                if agent_i == agent_j:
                    continue  # Skip checking against itself

                
                # Check if agent_i envies agent_j's allocation
                if utility_matrix[agent_i][agent_i] >= utility_matrix[agent_i][agent_j]:
                    continue  # No envy, EF1 condition satisfied for this pair
                else:
                    # Check if removing any single item from agent_j's allocation eliminates the envy
                    is_envy_eliminated = False
                    for item in np.where(allocation[agent_j] == 1)[0]:
                        if utility_matrix[agent_i][agent_i] >= utility_matrix[agent_i][agent_j] - self.valuation[agent_i][item]:
                            is_envy_eliminated = True
                            break  # If any condition satisfies the criterion, stop checking further

                    if not is_envy_eliminated:
                        return False  # Found a pair that violates EF1 condition

        return True
    
    def is_efx(self, allocation):
        """
        Check if the given allocation is EF1 (Envy-Free up to one item).

        Parameters:
        allocation (dict): A dictionary mapping each agent to their allocated items.
        valuations (dict): A nested dictionary where valuations[agent][item] is the valuation
                        of the item by the agent.

        Returns:
        bool: True if the allocation is EF1, False otherwise.
        """
        utility_matrix = np.dot(self.valuation, np.array(allocation).T)
        # Iterate through each pair of agents to check the EF1 condition
        for agent_i in self.agents:
            for agent_j in self.agents:
                if agent_i == agent_j:
                    continue  # Skip checking against itself

                
                # Check if agent_i envies agent_j's allocation
                if utility_matrix[agent_i][agent_i] >= utility_matrix[agent_i][agent_j]:
                    continue  # No envy, EF1 condition satisfied for this pair
                else:
                    # Check if removing any single item from agent_j's allocation eliminates the envy
                    is_envy_eliminated = False
                    for item in np.where(allocation[agent_j] == 1)[0]:
                        if utility_matrix[agent_i][agent_i] < utility_matrix[agent_i][agent_j] - self.valuation[agent_i][item]:
                            return False  # Found a pair that violates EFx condition

        return True
    

    def precompute_allocation_utilities(self):
        """
        Precompute the utilities of all possible allocations.
        """


        unique_allocations = self.all_allocations()
        for allocation in unique_allocations:
            allocation_tuple = self.allocation_to_tuple(allocation)
            self.allocation_utilities[allocation_tuple] = np.dot(self.valuation, np.array(allocation).T).diagonal()

    def allocation_to_tuple(self, allocation):
        """
        Convert an allocation to a hashable tuple format.
        """
        return tuple(tuple(agent_alloc) for agent_alloc in allocation)

    def is_pareto_optimal(self, proposed_allocation):
        """
        Check if the proposed allocation is Pareto optimal.

        Parameters:
        - proposed_allocation (list): The proposed allocation mapping each agent to a list of items.

        Returns:
        - bool, tuple: True and None if the allocation is Pareto optimal, otherwise False and the Pareto dominant allocation.
        """
        # Convert the proposed allocation to a tuple and calculate the utility for each agent
        proposed_allocation_tuple = self.allocation_to_tuple(proposed_allocation)
        proposed_utilities = np.dot(self.valuation, np.array(proposed_allocation).T).diagonal()

        # Check against precomputed Pareto optimal allocations first
        for po_allocation in self.pareto_optimal_utilities:
            po_utilities = self.allocation_utilities[po_allocation]

            better_off = False
            worse_off = False

            for agent in self.agents:
                if po_utilities[agent] > proposed_utilities[agent]:
                    better_off = True
                elif po_utilities[agent] < proposed_utilities[agent]:
                    worse_off = True
                    break

            if better_off and not worse_off:
                return False, po_allocation

        # Check against all precomputed allocations
        for allocation, allocation_utilities in self.allocation_utilities.items():
            better_off = False
            worse_off = False

            for agent in self.agents:
                if allocation_utilities[agent] > proposed_utilities[agent]:
                    better_off = True
                elif allocation_utilities[agent] < proposed_utilities[agent]:
                    worse_off = True
                    break

            if better_off and not worse_off:
                return False, allocation

        # If no better allocation is found, mark the proposed allocation as Pareto optimal
        self.pareto_optimal_utilities.append(proposed_allocation_tuple)
        return True, None


    def compute_efx_and_po_allocations(self):
        """
        Compute all allocations that are both Envy-Free up to one item (EF1) and Pareto Optimal (PO).

        Returns:
        list: A list of allocations that satisfy both EF1 and PO conditions.
        """
        
        # Generate all possible allocations
        all_allocations = self.all_allocations()
        
        ef1_po_allocations = []
        self.precompute_allocation_utilities()
        for allocation in all_allocations:
            # Check if the allocation is both EF1 and Pareto Optimal
            if self.is_efx(allocation):
                if self.is_pareto_optimal(allocation)[0]:
                    # Append the allocation to the list after converting it to a numpy array
                    ef1_po_allocations.append(np.array(allocation))

        return ef1_po_allocations
    
    def compute_ef1_and_po_allocations(self):
        """
        Compute all allocations that are both Envy-Free up to one item (EF1) and Pareto Optimal (PO).

        Returns:
        list: A list of allocations that satisfy both EF1 and PO conditions.
        """
        
        # Generate all possible allocations
        all_allocations = self.all_allocations()
        
        ef1_po_allocations = []
        self.precompute_allocation_utilities()
        for allocation in all_allocations:
            # Check if the allocation is both EF1 and Pareto Optimal
            if self.is_ef1(allocation):
                if self.is_pareto_optimal(allocation)[0]:
                    # Append the allocation to the list after converting it to a numpy array
                    ef1_po_allocations.append(np.array(allocation))

        return ef1_po_allocations


 


    

                                                     

        


    