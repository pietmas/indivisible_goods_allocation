from itertools import combinations
from functools import lru_cache




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



    def generate_distributions(self, n_items, n_agents):
        """
        Generate all possible distributions of n_items to n_agents in an optimized manner.

        This method recursively generates distributions by allocating a varying number of items to the first agent 
        and then distributing the remaining items among the remaining agents.

        Parameters:
        n_items (int): The total number of items to distribute.
        n_agents (int): The total number of agents to distribute items to.

        Yields:
        tuple: A tuple representing one possible distribution of items among the agents.
        """
        if n_agents == 1:
            # If there is only one agent, all items go to this agent
            yield (n_items,)
        else:
            # Iterate over possible number of items allocated to the first agent
            for first_agent_items in range(n_items + 1):
                # Recursively generate distributions for the remaining items and agents
                for rest in self.generate_distributions(n_items - first_agent_items, n_agents - 1):
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
                for subsequent_allocation in self.distribute_items_according_to_distribution(sorted(remaining_items_set), distribution[1:]):
                    
                    # Yield the current allocation combined with subsequent allocations
                    yield [tuple(sorted(items_for_first_agent))] + subsequent_allocation


    def all_allocations(self):
        """
        Generate all possible allocations of items to agents.

        Yields:
        dict: A dictionary where keys are agents and values are tuples of items allocated to them.
        """
        
        n_items, n_agents = self.nitems, self.nagents

        # Generate all possible ways to distribute the number of items to agents
        distributions = self.generate_distributions(n_items, n_agents)

        for distribution in distributions:
            # For each distribution, generate all unique sets of items for the agents
            for unique_set in self.distribute_items_according_to_distribution(self.items, distribution):
                # Create an allocation dictionary where each agent gets their corresponding bundle of items
                allocation = {agent: bundle for agent, bundle in zip(self.agents, unique_set)}
                
                # Yield the allocation
                yield allocation


    def is_ef1(self, allocation):
        """
        Check if the given allocation is EF1 (Envy-Free up to one item).

        Parameters:
        allocation (dict): A dictionary mapping each agent to their allocated items.

        Returns:
        bool: True if the allocation is EF1, False otherwise.
        """
        
        # Iterate through each pair of agents to check the EF1 condition
        for agent_i in self.agents:
            # Calculate the total valuation of agent_i for their own allocation
            valuation_i_for_own = sum(self.valuation[agent_i][item] for item in allocation[agent_i])
            
            for agent_j in self.agents:
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

                is_envy_eliminated = False

                # Check if removing any single item from agent_j's allocation eliminates the envy
                for item in allocation[agent_j]:
                    if valuation_i_for_own >= valuation_i_for_j - self.valuation[agent_i][item]:
                        is_envy_eliminated = True
                        break  # If any condition satisfies the criterion, stop checking further

                # If no single item removal can eliminate the envy, return False
                if not is_envy_eliminated:
                    return False  # Found a pair that violates EF1 condition
            
        return True  # All pairs satisfy EF1 condition

    def is_pareto_optimal(self, proposed_allocation):
        """
        Check if the proposed allocation is Pareto optimal.

        Parameters:
        proposed_allocation (dict): The proposed allocation mapping each agent to a list of items.

        Returns:
        bool, dict: True and None if the allocation is Pareto optimal, otherwise False and the Pareto dominant allocation.
        """
        
        # Calculate utility of proposed allocation for each agent
        proposed_utilities = {
            agent: self.compute_utility(proposed_allocation, agent)
            for agent in self.agents
        }
        
        # Generate all possible allocations
        unique_allocations = self.all_allocations()

        for allocation in unique_allocations:
            # Calculate utility of each unique allocation for each agent
            allocation_utilities = {
                agent: self.compute_utility(allocation, agent)
                for agent in self.agents
            }

            better_off = False
            worse_off = False
            
            # Compare utilities of the proposed allocation and the current allocation
            for agent in self.agents:
                if allocation_utilities[agent] > proposed_utilities[agent]:
                    better_off = True
                elif allocation_utilities[agent] < proposed_utilities[agent]:
                    worse_off = True
                    break

            # If there exists an allocation where at least one agent is better off and no agent is worse off,
            # the proposed allocation is not Pareto optimal
            if better_off and not worse_off:
                return False, allocation

        # No Pareto improvement found, indicating the proposed allocation is Pareto optimal
        return True, None


    def compute_ef1_and_po_allocations(self):
        """
        Compute all allocations that are both Envy-Free up to one item (EF1) and Pareto Optimal (PO).

        Returns:
        list: A list of allocations that satisfy both EF1 and PO conditions.
        """
        
        # Generate all possible allocations
        all_allocations = self.all_allocations()
        
        ef1_po_allocations = []

        for allocation in all_allocations:
            # Check if the allocation is both EF1 and Pareto Optimal
            if self.is_ef1(allocation):
                if self.is_pareto_optimal(allocation)[0]:
                    ef1_po_allocations.append(allocation)

        return ef1_po_allocations

    def compute_utility(self, allocation, agent):
        """
        Calculate the utility of a given allocation for a specific agent.

        Parameters:
        allocation (dict): A dictionary mapping each agent to their allocated items.
        agent (str): The agent for whom the utility is being calculated.

        Returns:
        int/float: The total utility value for the given agent based on the allocation.
        """
        # Helper function for memoization
        @lru_cache(maxsize=None)
        def utility(agent, allocation_tuple):
            return sum(self.valuation[agent][item] for item in allocation_tuple)

        # Convert allocation to a tuple of items for the specified agent
        allocation_tuple = tuple(allocation[agent])
        
        # Calculate and return the total utility for the given agent
        return utility(agent, allocation_tuple)


    