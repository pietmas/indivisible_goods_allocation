from itertools import permutations
from itertools import combinations, chain
from itertools import combinations, chain, permutations, product
from algorithms.brute_force import BruteForce
from algorithms.dict_version.barman import Barman
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

    def powerset(self, iterable):
        """
        Generate all non-empty subsets of the iterable.

        Parameters:
        iterable (iterable): An iterable (e.g., list, set, tuple) from which to generate the power set.

        Returns:
        iterator: An iterator that produces all non-empty subsets of the input iterable.
        """
        
        # Convert the input iterable to a list to ensure indexing and length operations can be performed
        s = list(iterable)
        
        # Use chain.from_iterable to flatten the list of combinations and generate all non-empty subsets
        # combinations(s, r) generates all possible combinations of the elements in s with length r
        # range(1, len(s)+1) ensures that r starts from 1 (excluding the empty subset) up to the length of s
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

    def remove_items_from_powerset(self, items):
        """
        Generate subsets of items by removing items in all possible combinations.

        Parameters:
        items (iterable): An iterable (e.g., list, set, tuple) from which to generate the subsets by removing elements.

        Returns:
        iterator: An iterator that produces all possible subsets of the input items, including the empty subset.
        """

        # Use chain.from_iterable to flatten the list of combinations and generate all possible subsets
        # combinations(items, r) generates all possible combinations of the elements in items with length r
        # range(len(items)) ensures that r starts from 0 (including the empty subset) up to len(items)-1
        return chain.from_iterable(combinations(items, r) for r in range(len(items)))

    def all_allocations(self, agents=None, items=None):
        """
        Generate all possible allocations of items to agents.

        Parameters:
        agents (iterable): A list or iterable of agents. If not provided, self.agents is used.
        items (iterable): A list or iterable of items. If not provided, self.items is used.

        Yields:
        dict: A dictionary representing a possible allocation.
            Each dictionary maps agents to their allocated bundle of items.
        """
        
        # Use self.agents if no agents are provided
        if agents is None:
            agents = self.agents
            
        # Use self.items if no items are provided
        if items is None:
            items = self.items

        # Generate all possible distributions of the items among the agents
        distributions = self.generate_distributions(len(items), len(agents))

        # Iterate over each possible distribution
        for distribution in distributions:
            # Generate all unique sets of items for the given distribution
            for unique_set in self.distribute_items_according_to_distribution(items, distribution):
                # Create an allocation dictionary mapping each agent to their corresponding bundle of items
                allocation = {agent: bundle for agent, bundle in zip(agents, unique_set)}
                # Yield the allocation
                yield allocation

    def generate_distributions(self, num_items, num_agents):
        """
        Generate all possible distributions of items to agents.

        Parameters:
        num_items (int): The total number of items to distribute.
        num_agents (int): The total number of agents to distribute items to.

        Yields:
        tuple: A tuple representing a possible distribution of items among agents.
        """
        
        # Use itertools.product to generate all combinations of item counts per agent
        # Each combination has num_agents elements, where each element can be in the range from 0 to num_items
        for combo in product(range(num_items + 1), repeat=num_agents):
            if sum(combo) == num_items:
                yield combo

    def distribute_items_according_to_distribution(self, items, distribution):
        """
        Generate all unique sets of items for each distribution.

        Parameters:
        items (iterable): The items to be distributed.
        distribution (tuple): A tuple representing the number of items each agent should receive.

        Yields:
        list: A list of tuples where each tuple is a unique set of items for an agent.
        """
        
        if len(distribution) == 1:
            # Only one agent, so they get all items
            yield [tuple(sorted(items))]
        else:
            # Number of items the first agent should receive
            first_agent_items = distribution[0]
            
            # Generate all combinations of items for the first agent
            for items_for_first_agent in combinations(items, first_agent_items):
                # Create a list of remaining items
                remaining_items = list(items)
                
                # Remove the items assigned to the first agent from the remaining items
                for item in items_for_first_agent:
                    remaining_items.remove(item)
                    
                # Recursively generate the distribution for the remaining agents
                for subsequent_allocation in self.distribute_items_according_to_distribution(remaining_items, distribution[1:]):
                    # Yield the allocation for the first agent combined with the subsequent allocation
                    yield [tuple(sorted(items_for_first_agent))] + subsequent_allocation


    def is_ef1_allocation(self, allocation=None, agents=None, items=None):
        """
        Check if the given allocation is EF1.

        Parameters:
        allocation (dict): A dictionary mapping each agent to their allocated items. If not provided, self.allocation is used.
        agents (iterable): A list or iterable of agents. If not provided, self.agents is used.
        items (iterable): A list or iterable of items. If not provided, self.items is used.

        Returns:
        tuple: A tuple containing a boolean indicating if the allocation is EF1 and 
            a tuple of agents (agent_i, agent_j) that violate the EF1 condition if the allocation is not EF1.
        """
        
        # Use self.allocation if no allocation is provided
        if allocation is None:
            allocation = self.allocation
            
        # Use self.agents if no agents are provided
        if agents is None:
            agents = self.agents
            
        # Use self.items if no items are provided
        if items is None:
            items = self.items

        # Iterate through each pair of agents to check the EF1 condition
        for agent_i in agents:
            # Calculate the total valuation of agent_i for their own allocation
            valuation_i_for_own = sum(self.valuation[agent_i][item] for item in allocation[agent_i])

            for agent_j in agents:
                if agent_i == agent_j:
                    continue  # Skip checking against itself

                # Calculate the total valuation of agent_i for agent_j's allocation
                valuation_i_for_j = sum(self.valuation[agent_i][item] for item in allocation[agent_j])
                
                # Calculate the total valuation of agent_j for their own allocation
                valuation_j_for_own = sum(self.valuation[agent_j][item] for item in allocation[agent_j])
                
                # Calculate the total valuation of agent_j for agent_i's allocation
                valuation_j_for_i = sum(self.valuation[agent_j][item] for item in allocation[agent_i])

                # Check if agent_i envies agent_j's allocation
                if valuation_i_for_own >= valuation_i_for_j:
                    continue  # No envy, EF1 condition satisfied for this pair

                # Check if removing any single item from agent_j's allocation eliminates the envy
                is_envy_eliminated = False
                for item in allocation[agent_j]:
                    if valuation_i_for_own >= valuation_i_for_j - self.valuation[agent_i][item]:
                        if valuation_j_for_i + self.valuation[agent_j][item] >= valuation_j_for_own:
                            is_envy_eliminated = True
                            break  # If any condition satisfies the criterion, stop checking further

                if not is_envy_eliminated:
                    return False, (agent_i, agent_j)  # Found a pair that violates EF1 condition

        return True, None  # All pairs satisfy the EF1 condition

    def is_pareto_optimal(self, allocation=None, agents=None, items=None):
        """
        Check if the proposed allocation is Pareto optimal.

        Parameters:
        allocation (dict): The proposed allocation mapping each agent to a list of items. If not provided, self.allocation is used.
        agents (list): A list of agent names. If not provided, self.agents is used.
        items (list): A list of item names. If not provided, self.items is used.

        Returns:
        tuple: A tuple containing a boolean indicating if the allocation is Pareto optimal and 
            the Pareto dominant allocation if it is not Pareto optimal.
        """
        
        # Use self.allocation if no allocation is provided
        if allocation is None:
            allocation = self.allocation
            
        # Use self.agents if no agents are provided
        if agents is None:
            agents = self.agents
            
        # Use self.items if no items are provided
        if items is None:
            items = self.items

        # Calculate utility of proposed allocation for each agent
        proposed_utilities = {
            agent: sum(self.valuation[agent][item] for item in allocation[agent]) if allocation[agent] else 0 
            for agent in agents
        }

        # Generate all possible unique allocations
        unique_allocations = self.all_allocations(agents=agents, items=items)

        # Iterate through all possible allocations
        for new_allocation in unique_allocations:
            # Calculate utility of the new allocation for each agent
            allocation_utilities = {
                agent: sum(self.valuation[agent][item] for item in new_allocation[agent])
                for agent in agents
            }

            better_off = False
            worse_off = False
            
            # Compare utilities of the proposed and new allocations
            for agent in agents:
                if allocation_utilities[agent] > proposed_utilities[agent]:
                    better_off = True
                elif allocation_utilities[agent] < proposed_utilities[agent]:
                    worse_off = True
                    break

            # Check for Pareto improvement
            if better_off and not worse_off:
                # Found a Pareto improvement, return False and the Pareto dominant allocation
                return False, new_allocation

        # No Pareto improvement found, return True and None to indicate the proposed allocation is Pareto optimal
        return True, None


    def is_resource_monotonic(self, removed_items):
        """
        Check if removing items keeps the allocation resource monotonic.

        Parameters:
        removed_items (list): A list of items to be removed from the allocation.

        Returns:
        tuple: A tuple containing a boolean indicating if the allocation is resource monotonic and 
            a list of non-monotonic allocations encountered during the check.
        """
        
        # Compute the items left after removing the specified items
        all_items = [item for sublist in self.allocation.values() for item in sublist if item not in removed_items]

        # Initialize an empty list to keep track of new allocations that have been visited
        visited_allocations = set()
        # Variable to store new allocations
        new_allocations = []

        # Compute the new allocations based on the method specified
        if self.method == "bf":
            new_allocations = self.compute_ef1_and_po_allocations(items=all_items)
        elif self.method == "mnw":
            new_allocations = self.maximize_nash_welfare(items=all_items)
        elif self.method == "fea":
            fea = Barman(self.n_agents, self.n_items, self.valuation, items=all_items)
            new_allocations, _ = fea.run_algorithm()
        elif self.method == "gen_adjusted_winner":
            aw = GeneralizedAdjustedWinner(self.valuation, items=all_items)
            new_allocations = aw.get_allocation()
        elif self.method == "adjusted_winner":
            aw = AdjustedWinner(self.valuation)
            new_allocations = aw.get_allocation(item_subset=all_items)

        # Ensure new_allocations is a list of dictionaries
        if isinstance(new_allocations, dict):
            new_allocations = [new_allocations]

        # Variable to track if the allocation is resource monotonic
        is_monotonic = False
        # List to store non-monotonic allocations
        non_monotonic_allocations = []

        # Iterate through each new allocation
        for new_allocation in new_allocations:
            # Convert the allocation to a frozenset of items for each agent to save visited allocations
            allocation_key = frozenset((agent, frozenset(new_allocation[agent])) for agent in new_allocation)
            
            # Skip if this allocation has been visited before
            if allocation_key in visited_allocations:
                continue
            
            # Mark this allocation as visited
            visited_allocations.add(allocation_key)

            # Assume this allocation maintains resource monotonicity until a decrease in utility is found
            decreases_utility = False

            # Check each agent's utility in the new allocation against their utility in the original allocation
            for agent in self.agents:
                original_utility = sum(self.valuation[agent][item] for item in self.allocation[agent]) if self.allocation[agent] else 0
                new_utility = sum(self.valuation[agent][item] for item in new_allocation[agent]) if new_allocation[agent] else 0

                # If any agent's utility increases, mark it as non-monotonic and break
                if new_utility > original_utility:
                    decreases_utility = True
                    non_monotonic_allocations.append(new_allocation)
                    break

            # If this allocation does not increase utility for any agent, resource monotonicity is maintained
            if not decreases_utility:
                is_monotonic = True
                return is_monotonic, non_monotonic_allocations

        # Return the result and the list of non-monotonic allocations
        return is_monotonic, non_monotonic_allocations


    def check_resource_monotonicity(self):
        """
        Check resource monotonicity for all possible combinations of item removals,
        including when the number of items is less than or equal to the number of agents.
        
        Returns:
        - (bool, list, list): A tuple where the first element is a boolean indicating if the
        allocation is resource monotonic, the second element is a list of item combinations
        that break resource monotonicity (if any), and the third is the allocations that break monotonicity.
        """
        
        # Initialize variables to track removed items and non-monotonic allocations
        removed_items = []
        non_monotonic_allocations = []
        res_monotonic = True
        
        # Check resource monotonicity for all combinations of item removals
        for removed_item_combination in self.remove_items_from_powerset(self.items):
            if removed_item_combination:  # Skip the case where no items are removed
                is_monotonic, na = self.is_resource_monotonic(list(removed_item_combination))
                if not is_monotonic:
                    res_monotonic = False
                    removed_items.append(removed_item_combination)
                    non_monotonic_allocations.append(na)
                    # Return immediately if you only want to find the first violation
                    return res_monotonic, removed_items, non_monotonic_allocations
        
        # Return the result and lists of violations if any
        return res_monotonic, removed_items if not res_monotonic else None, non_monotonic_allocations if not res_monotonic else None


                            
    def is_population_monotonic(self, removed_agents):
        """
        Check population monotonicity for the removal of a single agent by evaluating all possible new allocations.
        If at least one new allocation does not decrease the utility for any remaining agent, return True.

        Parameters:
        removed_agents (list): A list of agents to be removed from the allocation.

        Returns:
        tuple: A tuple containing a boolean indicating if the allocation is population monotonic and 
            a list of non-monotonic allocations encountered during the check.
        """
        
        # Recompute the list of agents excluding the removed agents
        agents = self.agents.copy()
        for removed_agent in removed_agents:
            agents.remove(removed_agent)

        # Initialize an empty set to keep track of visited new allocations
        visited_allocations = set()
        # Variable to store new allocations
        new_allocations = []

        # Compute the new allocations based on the method specified
        if self.method == "bf":
            new_allocations = self.compute_ef1_and_po_allocations(agents=agents)
        elif self.method == "mnw":
            new_allocations = self.maximize_nash_welfare(agents=agents)
        elif self.method == "fea":
            fea = Barman(self.n_agents, self.n_items, self.valuation, agents=agents)
            new_allocations, _ = fea.run_algorithm()

        # Ensure new_allocations is a list of dictionaries
        if isinstance(new_allocations, dict):
            new_allocations = [new_allocations]

        # Variable to track if the allocation is population monotonic
        is_monotonic = False
        # List to store non-monotonic allocations
        non_monotonic_allocations = []

        # Iterate through each new allocation
        for new_allocation in new_allocations:
            # Convert the allocation to a frozenset of items for each agent to save visited allocations
            allocation_key = frozenset((agent, frozenset(new_allocation[agent])) for agent in new_allocation)
            
            # Skip if this allocation has been visited before
            if allocation_key in visited_allocations:
                continue
            
            # Mark this allocation as visited
            visited_allocations.add(allocation_key)

            # Assume this allocation maintains population monotonicity until a decrease in utility is found
            decreases_utility = False

            # Check each agent's utility in the new allocation against their utility in the original allocation
            for agent in agents:
                original_utility = sum(self.valuation[agent][item] for item in self.allocation[agent])
                new_utility = sum(self.valuation[agent][item] for item in new_allocation[agent])

                # If any agent's utility decreases, mark it as non-monotonic and break
                if new_utility < original_utility:
                    decreases_utility = True
                    non_monotonic_allocations.append(new_allocation)
                    break

            # If this allocation does not decrease utility for any agent, population monotonicity is maintained
            if not decreases_utility:
                return True, non_monotonic_allocations

        # Return the result and the list of non-monotonic allocations
        return is_monotonic, non_monotonic_allocations


    def check_population_monotonicity(self):
        """
        Check population monotonicity for all possible combinations of agent removals,
        stopping when the number of remaining agents is strictly less than two.
        
        Returns:
        - (bool, list, list): A tuple where the first element is a boolean indicating if the
        allocation is population monotonic, the second element is a list of agent combinations
        that break population monotonicity (if any), and the third element is a list of
        non-monotonic allocations.
        """
        
        # Copy the list of agents
        agents = self.agents.copy()
        # Initialize lists to track agent combinations that break monotonicity and non-monotonic allocations
        non_monotonic_agent_combinations = []
        non_monotonic_allocations = []

        # Variable to track if all combinations maintain population monotonicity
        all_monotonic = True

        # Iterate through all possible subsets of agents using the powerset method
        for remove_agents in self.powerset(agents):
            if len(remove_agents) < len(agents) - 1:  # Skip the case where only one agent is removed
                is_monotonic, non_monotonic_allocation = self.is_population_monotonic(remove_agents)
                if not is_monotonic:
                    all_monotonic = False
                    non_monotonic_agent_combinations.append(remove_agents)
                    non_monotonic_allocations.append(non_monotonic_allocation)

                    # Return immediately if a violation is found
                    return all_monotonic, non_monotonic_agent_combinations, non_monotonic_allocations
                    
        # Return the result and lists of violations if any
        return all_monotonic, non_monotonic_agent_combinations, non_monotonic_allocations

        
    def check_resource_and_population_monotonicity(self):
        """
        Check resource monotonicity for all possible combinations of item removals,
        stopping when the number of items is strictly less than the number of agents.
        
        Parameters:
        - original_allocation: dict, mapping each agent to their allocated items.
        - utilities: dict, mapping each agent to a dict of items and their corresponding utility.
        
        Returns:
        - (bool, list): A tuple where the first element is a boolean indicating if the
        allocation is resource monotonic and the second element is a list of item combinations
        that break resource monotonicity, if any.
        """
        agents = self.agents.copy()
        items = self.items.copy()
        pop = []
        new_allocations = []
        removed_item = []
        res_monotonic = True
        for removed_item_combination in self.remove_items_from_powerset(items):
            if removed_item_combination and len(removed_item_combination) == 1:  # Skip the case where no items are removed
                for removed_agents in self.powerset(agents):
                    if len(removed_agents) < len(agents):
                        is_monotonic, na = self.is_population_monotonic(removed_agents)
                        if not is_monotonic:
                            res_monotonic = False
                            pop.append(removed_agents)
                            new_allocations.append(na)
                            removed_item.append(removed_item_combination)
                            break
        return res_monotonic, removed_item, new_allocations

    def compute_ef1_and_po_allocations(self, valuations=None, items=None, agents=None):
        """
        Compute allocations that are both Envy-Free up to one item (EF1) and Pareto Optimal (PO).

        Parameters:
        valuations (dict): A nested dictionary where valuations[agent][item] gives the utility of an item for an agent.
                        If not provided, self.valuation is used.
        items (list): A list of items to be allocated. If not provided, self.items is used.
        agents (list): A list of agents to whom items are allocated. If not provided, self.agents is used.

        Returns:
        list: A list of allocations that are both EF1 and PO.
        """
        
        # Use self.valuation if no valuations are provided
        if valuations is None:
            valuations = self.valuation
            
        # Use self.items if no items are provided
        if items is None:
            items = self.items
            
        # Use self.agents if no agents are provided
        if agents is None:
            agents = self.agents

        # Generate all possible allocations of items to agents
        all_allocations = self.all_allocations(agents=agents, items=items)
        # List to store allocations that are both EF1 and PO
        ef1_po_allocations = []

        # Iterate through each allocation to check if it is both EF1 and PO
        for allocation in all_allocations:
            # Check if the allocation is EF1 and PO
            is_ef1, _ = self.is_ef1_allocation(allocation=allocation, agents=agents, items=items)
            is_po, _ = self.is_pareto_optimal(allocation=allocation, agents=agents, items=items)
            
            if is_ef1 and is_po:
                ef1_po_allocations.append(allocation)

        # Return the list of allocations that are both EF1 and PO
        return ef1_po_allocations

                

    
    def compute_nash_welfare(self, allocation):
        """
        Compute the Nash Welfare for a given allocation.

        Parameters:
        allocation (dict): A dictionary mapping each agent to their allocated items.

        Returns:
        float: The Nash welfare of the given allocation.
        """
        
        # Initialize the Nash welfare to 1 (multiplicative identity)
        nw = 1
        
        # Iterate through each agent and their allocated items in the allocation
        for agent, items in allocation.items():
            # Calculate the total valuation of the items for the agent
            if not items:
                agent_valuation = 0  # If no items are allocated, the valuation is 0
            else:
                agent_valuation = sum(self.valuation[agent][item] for item in items)
            
            # Multiply the Nash welfare by the agent's valuation
            nw *= agent_valuation
        
        # Return the computed Nash welfare
        return nw


    def maximize_nash_welfare(self, items=None, agents=None):
        """
        Find the allocation that maximizes the Nash Welfare.

        Parameters:
        items (list): A list of items to be allocated. If not provided, self.items is used.
        agents (list): A list of agents to whom items are allocated. If not provided, self.agents is used.

        Returns:
        list: A list of allocations that maximize the Nash welfare.
        """
        
        # Use self.items if no items are provided
        if items is None:
            items = self.items
            
        # Use self.agents if no agents are provided
        if agents is None:
            agents = self.agents

        # Initialize variables to track the maximum Nash welfare and corresponding allocations
        max_nash_welfare = 0
        best_allocations = []

        # Generate all possible allocations of items to agents
        for allocation in self.all_allocations(agents=agents, items=items):
            # Compute the Nash welfare for the current allocation
            nash_welfare = self.compute_nash_welfare(allocation)

            # Update the maximum Nash welfare and best allocations if a better allocation is found
            if nash_welfare > max_nash_welfare:
                max_nash_welfare = nash_welfare
                best_allocations = [allocation]
            elif nash_welfare == max_nash_welfare:
                best_allocations.append(allocation)

        # Return the list of allocations that maximize the Nash welfare
        return best_allocations




                    
                    
                    
    
    