import numpy as np
from itertools import chain, combinations, permutations

class SeqMinEnvy:
    """
    This class implements an algorithm for sequentially allocate items, taking the PO allocation that respect RM.
    The allocation process is designed to ensure fairness by considering agents' valuations and Pareto optimality.

    Attributes:
    - n (int): Number of agents.
    - m (int): Number of items.
    - valuation (numpy.ndarray): Valuation matrix where valuation[i][j] represents the valuation of agent i for item j.
    - items (list): List of items represented by their indices.
    - agents (list): List of agents represented by their indices.
    - allocation (numpy.ndarray): 2D array to store the allocation result where allocation[i][j] = 1 if item j is allocated to agent i.

    Methods:
    - compute_utility_matrix(allocation_matrix): Computes the utility matrix based on the given allocation matrix.
    - envy(allocation): Computes the maximal envy between the agents based on the given allocation.
    - generate_distributions(n_items, n_agents): Generates all possible distributions of n_items to n_agents.
    - distribute_items_according_to_distribution(items, distribution): Generates all unique sets of items for each distribution.
    - all_allocations(num_total_items): Generates all possible allocations of items to agents.
    - compute_po_allocations(initial_allocation_matrix): Computes Pareto optimal allocations.
    - is_pareto_optimal(proposed_allocation): Checks if the proposed allocation is Pareto optimal.
    - trade_items_algorithm(): Distributes items to agents in a round-robin fashion considering their valuations.
    """

    def __init__(self, n_agents, n_items, valuations):
        """
        Initialize the SeqMinEnvy class.

        Parameters:
        n_agents (int): Number of agents.
        n_items (int): Number of items.
        valuations (np.ndarray): Valuation matrix where valuations[i][j] is the valuation of agent i for item j.
        """
        self.n = n_agents
        self.m = n_items
        self.valuation = valuations
        self.items = list(range(n_items))
        self.agents = list(range(n_agents))
        self.allocation = np.zeros((n_agents, n_items), dtype=int)

    def compute_utility_matrix(self, allocation_matrix):
        """
        Computes the utility matrix based on the given allocation matrix.

        Parameters:
        - allocation_matrix (numpy.ndarray): A 2D array where allocation_matrix[i][j] indicates if item j is allocated to agent i.

        Returns:
        - numpy.ndarray: The utility matrix where each element represents the utility an agent derives from the allocation.
        """
        return np.dot(self.valuation, allocation_matrix.T)
    
    def envy(self, allocation):
        """
        Compute the maximal envy between the agents.

        Parameters:
        - allocation (np.ndarray): A 2D numpy array where allocation[agent][item] is 1 if the item is allocated to the agent, and 0 otherwise.

        Returns:
        - tuple: A tuple containing:
            - numpy.ndarray: The sorted order of agents based on their envy.
            - numpy.ndarray: The maximal envy values for each agent.
        """
        utility_matrix = self.compute_utility_matrix(allocation)
        envy_matrix = np.zeros((self.n, self.n))
        for j in self.agents:        
            envy_matrix[j] = utility_matrix[j] - utility_matrix[j][j]
        
        agents_envy = np.max(envy_matrix, axis=1)
        return np.argsort(agents_envy), agents_envy

    def generate_distributions(self, n_items, n_agents):
        """
        Generate all possible distributions of n_items to n_agents.

        Parameters:
        - n_items (int): The total number of items to be distributed.
        - n_agents (int): The total number of agents among whom the items will be distributed.

        Yields:
        - tuple: A tuple representing one possible distribution of items among the agents.
            Each element in the tuple corresponds to the number of items assigned to an agent.
        """
        if n_agents == 1:
            yield (n_items,)
        else:
            for first_agent_items in range(n_items + 1):
                for rest in self.generate_distributions(n_items - first_agent_items, n_agents - 1):
                    yield (first_agent_items,) + rest

    def distribute_items_according_to_distribution(self, items, distribution):
        """
        Generate all unique sets of items for each distribution.

        Parameters:
        - items (list): The list of items to be distributed.
        - distribution (list): A list where each element represents the number of items each agent should receive.

        Yields:
        - list: A list of tuples where each tuple represents a unique set of items distributed according to the distribution.
        """
        if len(distribution) == 1:
            yield [tuple(items)]
        else:
            first_agent_items = distribution[0]
            items_set = set(items)
            for items_for_first_agent in combinations(items, first_agent_items):
                remaining_items_set = items_set - set(items_for_first_agent)
                for subsequent_allocation in self.distribute_items_according_to_distribution(
                        sorted(remaining_items_set), distribution[1:]):
                    yield [tuple(sorted(items_for_first_agent))] + subsequent_allocation

    def all_allocations(self, num_total_items):
        """
        Generate all possible allocations of items to agents.

        Yields:
        - numpy.ndarray: A matrix representation of the allocation where each row corresponds to an agent
                         and each column corresponds to an item. A value of 1 indicates that the item is
                         allocated to the agent, and 0 otherwise.
        """
        distributions = self.generate_distributions(self.m, self.n)
        current_items = list(range(num_total_items))
        for distribution in distributions:
            for unique_set in self.distribute_items_according_to_distribution(current_items, distribution):
                allocation_matrix = np.zeros((self.n, self.m), dtype=int)
                for agent_index, bundle in enumerate(unique_set):
                    for item in bundle:
                        item_index = self.items.index(item)
                        allocation_matrix[agent_index][item_index] = 1
                yield allocation_matrix

    def compute_po_allocations(self, initial_allocation_matrix):    
        """
        Compute Pareto optimal allocations.

        Parameters:
        - initial_allocation_matrix (numpy.ndarray): The initial allocation matrix.

        Returns:
        - list: A list of Pareto optimal allocations.
        """
        initial_utilities_matrix = self.compute_utility_matrix(initial_allocation_matrix)
        initial_utility = initial_utilities_matrix.diagonal()

        num_total_items = np.sum(np.sum(initial_allocation_matrix, axis=1)).astype(int) + 1
        all_allocation = self.all_allocations(num_total_items)
        po_allocations = []

        for allocation in all_allocation:
            utility_matrix = self.compute_utility_matrix(allocation)
            utility = utility_matrix.diagonal()
            if self.is_pareto_optimal(allocation)[0] and np.all(utility >= initial_utility):
                po_allocations.append(np.array(allocation))

        return po_allocations

    def is_pareto_optimal(self, proposed_allocation):
        """
        Check if the proposed allocation is Pareto optimal.

        Parameters:
        - proposed_allocation (numpy.ndarray): The proposed allocation matrix.

        Returns:
        - tuple: A tuple containing:
            - bool: True if the allocation is Pareto optimal, False otherwise.
            - numpy.ndarray or None: The Pareto dominant allocation if one exists, otherwise None.
        """
        proposed_utilities = self.compute_utility_matrix(proposed_allocation).diagonal()
        total_items = np.sum(np.sum(proposed_allocation, axis=1)).astype(int)
        unique_allocations = self.all_allocations(total_items)

        for allocation in unique_allocations:
            allocation_utilities = self.compute_utility_matrix(allocation).diagonal()

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

        return True, None

    def trade_items_algorithm(self):
        """
        Distribute n_items to n_agents in a round-robin fashion considering their valuations.

        Returns:
        - numpy.ndarray: The final allocation matrix indicating which agent receives which items.
        """
        unallocated_items = self.items[:]
        valuation_t = [0, 0]

        # Initial item selection
        preferred_item = unallocated_items[0]
        agent = np.argmax(self.valuation[:, preferred_item])
        self.allocation[agent][preferred_item] = 1
        unallocated_items.remove(preferred_item)

        while unallocated_items:
            preferred_item = unallocated_items[0]
            po_allocation = self.compute_po_allocations(self.allocation)
            minimal_envy = np.inf

            for po in po_allocation:
                order, envy_matrix = self.envy(po)
                maximal_envy = np.max(envy_matrix)

                if maximal_envy < minimal_envy:
                    minimal_envy = maximal_envy
                    optimal_allocation = po

            self.allocation = optimal_allocation
            unallocated_items = unallocated_items[1:]

        return self.allocation
