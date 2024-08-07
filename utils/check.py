from itertools import permutations
from itertools import combinations, chain
from itertools import combinations, chain, permutations, product
from algorithms.brute_force import BruteForce
from algorithms.garg import GargAlgorithm
from algorithms.barman import Barman
from algorithms.generalized_adjusted_winner import GeneralizedAdjustedWinner
from algorithms.dict_version.adjustedwinner import AdjustedWinner
import numpy as np

class Checker:
    def __init__(self, n_agents, n_items, valuation, allocation, method = "bf"):
        self.n_items = n_items
        self.n_agents = n_agents
        self.items = list(range(n_items))
        self.agents = list(range(n_agents))
        self.valuation = valuation
        self.allocation = allocation
        self.method = method
        self.allocation_utilities = {}
        self.pareto_optimal_utilities = {}

    def powerset(self, iterable):
        """
        Generate all non-empty subsets of the given iterable.

        Args:
            iterable (iterable): An iterable such as a list, tuple, or set.

        Returns:
            itertools.chain: A chain object containing all non-empty subsets of the input iterable.
        """
        # Convert the input iterable into a list to ensure it is indexable
        s = list(iterable)
        
        # Use chain and combinations to generate all subsets of the iterable
        # combinations(s, r) generates all combinations of the elements in s of length r
        # chain.from_iterable flattens the list of combinations into a single iterable
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


    def remove_items_from_powerset(self, items):
        """
        Generate subsets of the given items by removing items in all possible combinations.

        Args:
            items (iterable): An iterable such as a list, tuple, or set.

        Returns:
            itertools.chain: A chain object containing all subsets of the input items,
                            including the empty set.
        """
        # Use chain and combinations to generate all subsets of the items
        # combinations(items, r) generates all combinations of the elements in items of length r
        # chain.from_iterable flattens the list of combinations into a single iterable
        # This includes the empty combination (removing all items) by including range(len(items))
        return chain.from_iterable(combinations(items, r) for r in range(len(items)))

    def generate_distributions(self, n_items, n_agents):
        """
        Generate all possible distributions of n_items to n_agents in an optimized manner.

        Args:
            n_items (int): The total number of items to distribute.
            n_agents (int): The total number of agents to distribute items to.

        Yields:
            tuple: A tuple representing a possible distribution of items among agents, 
                where each element in the tuple corresponds to the number of items 
                assigned to each agent.
        """
        
        # Base case: If there is only one agent, yield a tuple with all items assigned to that agent
        if n_agents == 1:
            yield (n_items,)
        else:
            # Iterate over the number of items the first agent can take, from 0 to n_items
            for first_agent_items in range(n_items + 1):
                # Recursively generate distributions for the remaining items and agents
                for rest in self.generate_distributions(n_items - first_agent_items, n_agents - 1):
                    # Yield the current distribution with the items for the first agent 
                    # prepended to the rest of the distribution
                    yield (first_agent_items,) + rest


    def distribute_items_according_to_distribution(self, items, distribution):
        """
        Generate all unique sets of items for each agent according to the given distribution.

        Args:
            items (iterable): An iterable of items to be distributed.
            distribution (tuple): A tuple representing the distribution of items among agents,
                                where each element indicates the number of items assigned to each agent.

        Yields:
            list of tuples: A list where each tuple represents the items assigned to an agent.
        """
        # Base case: If there is only one agent, yield all items as a single tuple
        if len(distribution) == 1:
            yield [tuple(items)]
        else:
            # Number of items to be assigned to the first agent
            first_agent_items = distribution[0]
            # Convert items to a set for efficient difference operations
            items_set = set(items)
            # Generate all possible combinations of items for the first agent
            for items_for_first_agent in combinations(items, first_agent_items):
                # Determine the remaining items after assigning to the first agent
                remaining_items_set = items_set - set(items_for_first_agent)
                # Recursively generate distributions for the remaining items and agents
                for subsequent_allocation in self.distribute_items_according_to_distribution(
                    sorted(remaining_items_set), distribution[1:]):
                    # Yield the current distribution with the items for the first agent 
                    # followed by the distribution for the remaining agents
                    yield [tuple(sorted(items_for_first_agent))] + subsequent_allocation

    def all_allocations(self, agents=None, items=None):
        """
        Generate all possible allocations of items to agents.

        Args:
            agents (list): A list of agents to allocate items to. If None, uses self.agents.
            items (list): A list of items to be allocated. If None, uses self.items.

        Yields:
            numpy.ndarray: A matrix representing an allocation where rows correspond to agents
                        and columns correspond to items. A value of 1 indicates the item is 
                        allocated to the agent, and 0 otherwise.
        """
        # If no agents are provided, use the default agents from the instance
        if agents is None:
            agents = self.agents
        # If no items are provided, use the default items from the instance
        if items is None:
            items = self.items

        # Determine the number of items and agents
        n_items = len(items)
        n_agents = len(agents)
        
        # Generate all possible distributions of items to agents
        distributions = self.generate_distributions(n_items, n_agents)

        # Iterate over each distribution
        for distribution in distributions:
            # Generate unique sets of items according to the current distribution
            for unique_set in self.distribute_items_according_to_distribution(items, distribution):
                # Initialize an allocation matrix with zeros
                allocation_matrix = np.zeros((self.n_agents, self.n_items), dtype=int)

                # Iterate over agents and their corresponding item bundles
                for i, bundle in zip(agents, unique_set):
                    # Mark the allocated items in the matrix
                    for item in bundle:
                        allocation_matrix[i][item] = 1

                # Yield the current allocation matrix
                yield allocation_matrix


    def is_ef1_allocation(self, valuation=None, allocation=None, agents=None, items=None):
        """
        Check if the given allocation is envy-free up to one item (EF1) for all agents.

        Args:
            valuation (numpy.ndarray): A matrix where valuation[i][j] represents the value
                                    agent i assigns to item j. If None, uses self.valuation.
            allocation (numpy.ndarray): A matrix where allocation[i][j] is 1 if item j is
                                        allocated to agent i, and 0 otherwise. If None, uses self.allocation.
            agents (list): A list of agents. If None, uses self.agents.
            items (list): A list of items. If None, uses self.items.

        Returns:
            bool: True if the allocation is EF1 for all agents, False otherwise.
        """
        if allocation is None:
            allocation = self.allocation
        if agents is None:
            agents = self.agents
        if items is None:
            items = self.items
        if valuation is None:
            valuation = self.valuation

        # Iterate through each pair of agents to check the EF1 condition
        for agent_i in agents:
            # Calculate the total valuation of agent_i for their own allocation
            valuation_i_for_own = np.dot(valuation[agent_i], allocation[agent_i])
            for agent_j in agents:
                if agent_i == agent_j:
                    continue  # Skip checking against itself
                
                # Calculate the total valuation of agent_i for agent_j's allocation
                valuation_i_for_j = np.dot(valuation[agent_i], allocation[agent_j])
                
                # Check if agent_i envies agent_j's allocation
                if valuation_i_for_own >= valuation_i_for_j:
                    continue  # No envy, EF1 condition satisfied for this pair
                
                is_envy_eliminated = False
                
                # Check if removing any single item from agent_j's allocation
                # would eliminate the envy from agent_i
                for item in np.where(allocation[agent_j] == 1)[0]:
                    if valuation_i_for_own >= valuation_i_for_j - valuation[agent_i][item]:
                        is_envy_eliminated = True
                        break  # If any condition satisfies the criterion, stop checking further

                if not is_envy_eliminated:
                    return False  # Found a pair that violates the EF1 condition

        return True
    
    def is_efx_allocation(self, valuation=None, allocation=None, agents=None, items=None):
        """
        Check if the given allocation is envy-free up to any item (EFX) for all agents.

        Args:
            valuation (numpy.ndarray): A matrix where valuation[i][j] represents the value
                                    agent i assigns to item j. If None, uses self.valuation.
            allocation (numpy.ndarray): A matrix where allocation[i][j] is 1 if item j is
                                        allocated to agent i, and 0 otherwise. If None, uses self.allocation.
            agents (list): A list of agents. If None, uses self.agents.
            items (list): A list of items. If None, uses self.items.

        Returns:
            bool: True if the allocation is EFX for all agents, False otherwise.
        """
        if allocation is None:
            allocation = self.allocation
        if agents is None:
            agents = self.agents
        if items is None:
            items = self.items
        if valuation is None:
            valuation = self.valuation

        # Iterate through each pair of agents to check the EFX condition
        for agent_i in agents:
            # Calculate the total valuation of agent_i for their own allocation
            valuation_i_for_own = np.dot(valuation[agent_i], allocation[agent_i])
            for agent_j in agents:
                if agent_i == agent_j:
                    continue  # Skip checking against itself
                
                # Calculate the total valuation of agent_i for agent_j's allocation
                valuation_i_for_j = np.dot(valuation[agent_i], allocation[agent_j])
                
                # Check if agent_i envies agent_j's allocation
                if valuation_i_for_own >= valuation_i_for_j:
                    continue  # No envy, EFX condition satisfied for this pair
                
                # Check if removing any single item from agent_j's allocation
                # would eliminate the envy from agent_i
                for item in np.where(allocation[agent_j] == 1)[0]:
                    if valuation_i_for_own < valuation_i_for_j - valuation[agent_i][item]:
                        return False  # Found a pair that violates the EFX condition

        return True

    def precompute_allocation_utilities(self, agents=None, items=None):
        """
        Precompute the utilities of all possible allocations for given agents and items.

        Args:
            agents (list): A list of agents. If None, uses self.agents.
            items (list): A list of items. If None, uses self.items.
        """
        if agents is None:
            agents = self.agents
        if items is None:
            items = self.items
        
        # Initialize dictionaries to store allocation utilities and pareto optimal utilities
        self.allocation_utilities = {}
        self.pareto_optimal_utilities = {}
        
        # Generate all unique allocations for the given agents and items
        unique_allocations = self.all_allocations(agents=agents, items=items)
        
        # Iterate over each unique allocation
        for allocation in unique_allocations:
            # Convert the allocation matrix to a tuple for use as a dictionary key
            allocation_tuple = self.allocation_to_tuple(allocation)
            
            # Calculate the utility of the allocation for each agent
            self.allocation_utilities[allocation_tuple] = np.dot(self.valuation, np.array(allocation).T).diagonal()

    def allocation_to_tuple(self, allocation):
        """
        Convert an allocation matrix to a hashable tuple format.

        Args:
            allocation (numpy.ndarray or list of lists): The allocation matrix where each row
                                                        represents the allocation for an agent.

        Returns:
            tuple: A tuple of tuples representing the allocation for each agent, making it hashable.
        """
        # Convert each agent's allocation (row in the matrix) to a tuple
        # Then convert the entire allocation to a tuple of these tuples
        return tuple(tuple(agent_alloc) for agent_alloc in allocation)

    def is_pareto_optimal(self, valuation=None, allocation=None, agents=None, items=None):
        """
        Check if the proposed allocation is Pareto optimal.

        Args:
            valuation (numpy.ndarray): A matrix where valuation[i][j] represents the value
                                    agent i assigns to item j. If None, uses self.valuation.
            allocation (numpy.ndarray): The proposed allocation matrix where each row represents an agent 
                                        and each column represents an item. If None, uses self.allocation.
            agents (list): A list of agents. If None, uses self.agents.
            items (list): A list of items. If None, uses self.items.

        Returns:
            bool: True if the allocation is Pareto optimal, False otherwise.
            tuple: The allocation that proves the proposed allocation is not Pareto optimal, or None.
        """
        if allocation is None:
            allocation = self.allocation  # Assuming self.allocation is already a NumPy matrix
        if agents is None:
            agents = self.agents
        if items is None:
            items = self.items
        if valuation is None:
            valuation = self.valuation

        # Convert the proposed allocation to a hashable tuple format
        proposed_allocation_tuple = self.allocation_to_tuple(allocation)
        # Calculate the utilities for the proposed allocation
        proposed_utilities = np.dot(self.valuation, np.array(allocation).T).diagonal()

        # Check against precomputed Pareto optimal allocations first
        for po_allocation in self.pareto_optimal_utilities.keys():
            po_utilities = self.pareto_optimal_utilities[po_allocation]

            better_off = False
            worse_off = False

            # Compare the utilities of the proposed allocation with the precomputed Pareto optimal allocation
            for agent in agents:
                if po_utilities[agent] > proposed_utilities[agent]:
                    better_off = True
                elif po_utilities[agent] < proposed_utilities[agent]:
                    worse_off = True
                    break

            # If there's a better off allocation with no worse off, the proposed allocation is not Pareto optimal
            if better_off and not worse_off:
                return False, po_allocation

        # Check against all precomputed allocations
        for allocation, allocation_utilities in self.allocation_utilities.items():
            better_off = False
            worse_off = False

            # Compare the utilities of the proposed allocation with each precomputed allocation
            for agent in agents:
                if allocation_utilities[agent] > proposed_utilities[agent]:
                    better_off = True
                elif allocation_utilities[agent] < proposed_utilities[agent]:
                    worse_off = True
                    break

            # If there's a better off allocation with no worse off, the proposed allocation is not Pareto optimal
            if better_off and not worse_off:
                return False, allocation

        # If no better allocation is found, mark the proposed allocation as Pareto optimal
        self.pareto_optimal_utilities[proposed_allocation_tuple] = proposed_utilities
        return True, None

    
    
    def is_resource_monotonic(self, removed_items):
        """
        Check if removing items keeps the allocation resource monotonic.

        Args:
            removed_items (list): A list of items to be removed from the allocation.

        Returns:
            bool: True if the allocation remains resource monotonic after removing items, False otherwise.
            list: If not resource monotonic, a list of new allocations that do not satisfy resource monotonicity.
                If resource monotonic, returns None.
        """
        # Compute the allocation with the items removed
        all_items = [item for item in self.items if item not in removed_items]
        
        # Generate new allocations based on the method specified
        if self.method == "bf":
            new_allocations = self.compute_ef1_and_po_allocations(items=all_items)
        elif self.method == "mnw":
            new_allocations = self.maximize_nash_welfare(items=all_items)
        elif self.method == "fea":
            fea = GargAlgorithm(self.n_agents, self.n_items, self.valuation, items=all_items)
            new_allocations, _ = fea.run_algorithm()
        elif self.method == "gen_adjusted_winner":
            aw = GeneralizedAdjustedWinner(self.valuation, items=all_items)
            new_allocations = aw.get_allocation()
        elif self.method == "adjusted_winner":
            aw = AdjustedWinner(self.valuation)
            new_allocations = aw.get_allocation(item_subset=all_items)
        
        is_monotonic = False
        na = []

        # Ensure new_allocations is a list
        if not isinstance(new_allocations, list):
            new_allocations = [new_allocations]

        # Calculate the original utility for each agent
        original_utility = np.dot(self.valuation, self.allocation.T).diagonal()
        
        # Iterate through each new allocation
        for new_allocation in new_allocations:
            # Calculate the new utility for each agent in the new allocation
            new_utility = np.dot(self.valuation, new_allocation.T).diagonal()
            
            # Check if any agent's utility decreases
            if np.all(new_utility <= original_utility):
                return True, None  # Found at least one allocation that satisfies resource monotonicity
            else:
                na.append(new_allocation)
        
        return is_monotonic, na if not is_monotonic else None

    def check_resource_monotonicity(self):
        """
        Check resource monotonicity for all possible combinations of item removals,
        including when the number of items is less than or equal to the number of agents.
        
        Returns:
            tuple: A tuple where the first element is a boolean indicating if the allocation is resource monotonic,
                the second element is a list of item combinations that break resource monotonicity (if any),
                and the third is the allocations that break monotonicity.
        """
        removed_item = []
        new_allocations = []
        res_monotonic = True

        # Check resource monotonicity for all combinations of item removals
        for removed_item_combination in self.remove_items_from_powerset(self.items):
            if removed_item_combination:  # Skip the case where no items are removed
                is_monotonic, na = self.is_resource_monotonic(list(removed_item_combination))
                if not is_monotonic:
                    res_monotonic = False
                    removed_item.append(removed_item_combination)
                    new_allocations.append(na)
                    break  # Stop checking further if a violation is found

        return res_monotonic, removed_item if not res_monotonic else None, new_allocations if not res_monotonic else None

                            
    def is_population_monotonic(self, removed_agents):
        """
        Check population monotonicity for the removal of a single agent by evaluating all possible new allocations.
        If at least one new allocation does not decrease the utility for any remaining agent, return True.

        Args:
            removed_agents (list): A list of agents to be removed from the allocation.

        Returns:
            bool: True if the allocation remains population monotonic after removing the agents, False otherwise.
            list: If not population monotonic, a list of new allocations that do not satisfy population monotonicity.
                If population monotonic, returns None.
        """
        # Recompute allocation without the removed agents
        agents = self.agents.copy()
        for removed_agent in removed_agents:
            agents.remove(removed_agent)
    
        # Generate new allocations based on the method specified
        if self.method == "bf":
            new_allocations = self.compute_ef1_and_po_allocations(agents=agents)
        elif self.method == "mnw":
            new_allocations = self.maximize_nash_welfare(agents=agents)
        elif self.method == "fea":
            fea = Barman(self.n_agents, self.n_items, self.valuation, agents=agents)
            new_allocations, _ = fea.run_algorithm()

        is_monotonic = False
        na = []

        # Ensure new_allocations is a list
        if isinstance(new_allocations, dict):
            new_allocations = [new_allocations]

        # Calculate the original utility for each agent
        original_utility = np.dot(self.valuation, self.allocation.T).diagonal()

        # Iterate through each new allocation
        for new_allocation in new_allocations:
            # Calculate the new utility for each agent in the new allocation
            new_utility = np.dot(self.valuation, new_allocation.T).diagonal()

            # Check if any remaining agent's utility decreases
            if np.all(new_utility[agents] >= original_utility[agents]):
                return True, None  # Found at least one allocation that satisfies population monotonicity
            else:
                na.append(new_allocation)
        
        return is_monotonic, na

    def check_population_monotonicity(self):
        """
        Check population monotonicity for all possible combinations of agent removals,
        stopping when the number of remaining agents is strictly less than the number of original agents.

        Returns:
            tuple: A tuple where the first element is a boolean indicating if the allocation is population monotonic,
                the second element is a list of agent combinations that break population monotonicity (if any),
                and the third is a list of new allocations that break monotonicity.
        """
        agents = self.agents.copy()
        pop = []
        new_allocations = []

        all_monotonic = True

        # Check population monotonicity for all combinations of agent removals
        for remove_agents in self.powerset(agents):
            if len(remove_agents) < len(agents):  # Skip the full set of agents
                is_monotonic, na = self.is_population_monotonic(remove_agents)
                
                if not is_monotonic:
                    all_monotonic = False
                    pop.append(remove_agents)
                    new_allocations.append(na)
                    break  # Stop checking further if a violation is found

        if not all_monotonic:
            return all_monotonic, pop, new_allocations
        else:
            return True, [], []


    def compute_ef1_and_po_allocations(self, valuations=None, items=None, agents=None):
        """
        Compute allocations that are both Envy-Free up to one item (EF1) and Pareto Optimal (PO).

        Args:
            valuations (numpy.ndarray): A matrix where valuation[i][j] represents the value
                                        agent i assigns to item j. If None, uses self.valuation.
            items (list): A list of items to be allocated. If None, uses self.items.
            agents (list): A list of agents. If None, uses self.agents.

        Returns:
            list: A list of numpy arrays where each array represents an allocation that is both EF1 and PO.
        """
        if valuations is None:
            valuations = self.valuation
        if items is None:
            items = self.items
        if agents is None:
            agents = self.agents

        # Generate all possible allocations
        all_allocations = self.all_allocations(agents=agents, items=items)

        # List to store allocations that are both EF1 and PO
        ef1_po_allocations = []

        # Precompute allocation utilities for efficiency
        self.precompute_allocation_utilities(agents=agents, items=items)

        # Iterate through all allocations
        for allocation in all_allocations:
            # Check if the allocation is EF1
            if self.is_ef1_allocation(allocation=allocation, agents=agents, items=items):
                # Check if the allocation is Pareto Optimal
                if self.is_pareto_optimal(allocation=allocation, agents=agents, items=items)[0]:
                    # Append the allocation to the list after converting it to a numpy array
                    ef1_po_allocations.append(np.array(allocation))

        return ef1_po_allocations

    
    def compute_efx_and_po_allocations(self, valuations=None, items=None, agents=None):
        """
        Compute allocations that are both Envy-Free up to any item (EFX) and Pareto Optimal (PO).

        Args:
            valuations (numpy.ndarray): A matrix where valuation[i][j] represents the value
                                        agent i assigns to item j. If None, uses self.valuation.
            items (list): A list of items to be allocated. If None, uses self.items.
            agents (list): A list of agents. If None, uses self.agents.

        Returns:
            list: A list of numpy arrays where each array represents an allocation that is both EFX and PO.
        """
        if valuations is None:
            valuations = self.valuation
        if items is None:
            items = self.items
        if agents is None:
            agents = self.agents

        # Generate all possible allocations
        all_allocations = self.all_allocations(agents=agents, items=items)

        # List to store allocations that are both EFX and PO
        efx_po_allocations = []

        # Precompute allocation utilities for efficiency
        self.precompute_allocation_utilities(agents=agents, items=items)

        # Iterate through all allocations
        for allocation in all_allocations:
            # Check if the allocation is EFX
            if self.is_efx_allocation(allocation):
                # Check if the allocation is Pareto Optimal
                if self.is_pareto_optimal(allocation)[0]:
                    # Append the allocation to the list after converting it to a numpy array
                    efx_po_allocations.append(np.array(allocation))

        return efx_po_allocations
        
        
    def compute_utilities(self, allocation):
        """
        Calculate utilities for each agent given an allocation.

        Args:
            allocation (numpy.ndarray): The allocation matrix where each row represents an agent 
                                        and each column represents an item.

        Returns:
            numpy.ndarray: A 1D array where each element represents the utility of an agent.
        """
        return np.dot(self.valuation, allocation.T).diagonal()
    
    def compute_nash_welfare(self, allocation):
        """
        Compute the Nash Welfare for a given allocation.

        Args:
            allocation (numpy.ndarray): The allocation matrix where each row represents an agent 
                                        and each column represents an item.

        Returns:
            float: The Nash Welfare of the given allocation.
        """
        # Calculate the utility for each agent
        utility = np.dot(self.valuation, allocation.T).diagonal()

        # Compute the Nash Welfare by taking the product of the utilities
        nw = np.prod(utility) 
        return nw

    def maximize_nash_welfare(self, items=None, agents=None):
        """
        Find the allocation that maximizes the Nash Welfare.

        Args:
            items (list): A list of items to be allocated. If None, uses self.items.
            agents (list): A list of agents. If None, uses self.agents.

        Returns:
            list: A list of allocations that maximize the Nash Welfare.
        """
        if items is None:
            items = self.items
        if agents is None:
            agents = self.agents

        max_nw = 0
        best_allocations = []

        # Iterate through all possible allocations
        for allocation in self.all_allocations(agents=agents, items=items):
            # Compute the Nash Welfare for the current allocation
            nw = self.compute_nash_welfare(allocation)

            # Update the best allocations if a higher Nash Welfare is found
            if nw > max_nw:
                max_nw = nw
                best_allocations = [allocation]
            elif nw == max_nw:
                best_allocations.append(allocation)

        return best_allocations
    

    
    

    
    