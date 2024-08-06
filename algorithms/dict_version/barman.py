import numpy as np
import math
import sys
from functools import lru_cache

class Barman:
    """
    Class to implement an algorithm presented in https://arxiv.org/abs/1707.04731.

    This class provides methods to compute envy-free up to one item (EF1) and pareto-optimal (PO) allocations 
    based on the algorithm presented in the paper:
    "Finding Fair and Efficient Allocations"
    by Siddharth Barman, Sanath Kumar Krishnamurthy, Rohit Vaish.
    Link: https://arxiv.org/abs/1707.04731.
    """
    def __init__(self, n, m, valuation, agent=None, item=None, epsilon=None):
        self.nagents = n
        self.nitems = m
        self.valuation = valuation
        self.agent = self.initialize_agent(agent)  # List of agents
        self.item = self.initialize_item(item)  # List of items   
        self.epsilon = self.initialize_epsilon()  # ε
        self.x = None  # Allocation x
        self.p = None  # Price vector p
        self.hierarchy = None  # Hierarchy of agents
        # Initialize more attributes as needed

    def initialize_agent(self, agent):
        """
        Initializes the agent(s) for the class instance.

        Parameters:
        agent (int or list of int): An agent ID or a list of agent IDs to initialize. 
                                    If None, it initializes all agents in the range from 0 to nagents-1.

        Returns:
        list of int: A list of agent IDs.
        """
        # Check if the input agent is None
        if agent is None:
            # If agent is None, return a list of all agent IDs from 0 to nagents-1
            return [i for i in range(self.nagents)]
        else:
            # If agent is not None, return the input agent
            return agent
        
    def initialize_item(self, item):
        """
        Initializes the item(s) for the class instance.

        Parameters:
        item (int or list of int): An item ID or a list of item IDs to initialize.
                                   If None, it initializes all items in the range from 0 to nitems-1.

        Returns:
        list of int: A list of item IDs.
        """
        # Check if the input item is None
        if item is None:
            # If item is None, return a list of all item IDs from 0 to nitems-1
            return [j for j in range(self.nitems)]
        else:
            # If item is not None, return the input item
            return item
        
        
    def initialize_epsilon(self):
        """
        Initializes the epsilon value based on the valuation matrix.

        Epsilon is calculated using the formula ε = 1 / (14 * m^3 * v_max^4), 
        where m is the number of items and v_max is the maximum valuation in the valuation matrix.

        Returns:
        float: The calculated epsilon value.

        Raises:
        ValueError: If the valuation matrix contains only zeros.
        """
        # Find the maximum valuation in the valuation matrix
        v_max = max(max(row) for row in self.valuation)
        epsilon = 0
        # Number of items
        m = self.nitems
        # Check if the maximum valuation is zero
        if v_max == 0:
            # Raise an error if the valuation matrix contains only zeros
            raise ValueError("Valuation matrix contains only zeros")
        else:
            # Calculate epsilon using the formula ε = 1 / (14 * m^3 * v_max^4)
            epsilon = 1 / (14 * (m ** 3) * (v_max ** 4))
        return epsilon
        
    def epsilon_round_valuations(self):
        """
        Adjust each valuation in the valuation matrix to be a multiple of epsilon (ε).

        This method iterates through each agent and item, adjusting their respective valuations to the nearest
        multiple of (1 + ε) based on a logarithmic rounding formula. This ensures that the valuations are scaled
        and quantized in a consistent manner, potentially useful for simplifying calculations and comparisons.

        Returns:
        - None: This method modifies the self.valuation matrix in place and does not return any value.
        """
        # Uncomment the following line to print the value of epsilon for debugging purposes
        # print("epsilon: ", self.epsilon)

        # Iterate over each agent
        for i in self.agent:
            # Iterate over each item
            for j in self.item:
                # Ensure ε is positive and the current valuation is positive
                if self.epsilon > 0 and self.valuation[i][j] > 0:
                    # Round each valuation to the nearest multiple of (1 + ε)
                    self.valuation[i][j] = (1 + self.epsilon) ** np.ceil(np.log(self.valuation[i][j]) / np.log(1 + self.epsilon))
                # If the valuation is zero, explicitly set it to zero (though it's already zero)
                elif self.valuation[i][j] == 0:
                    self.valuation[i][j] = 0


    def welfare_maximizing_allocation(self):
        """
        Compute the welfare-maximizing allocation of items to agents.

        This method allocates each item to the agent who values it the most, aiming to maximize the overall welfare.
        It initializes the allocation and price vector, then iterates through each item to find the agent with the
        highest valuation for that item. The item is then allocated to this agent, and its price is optionally set
        based on the highest valuation.

        
        Returns:
        - self.x: A dictionary where the keys are agent indices and the values are lists of item indices allocated to each agent.
        - self.p: A list where each index represents an item and the value is the price of that item.
        """
        # Initialize the allocation as a dictionary where each key is an agent and the value is a list of allocated items
        self.x = {agent: [] for agent in range(self.nagents)}
        # Initialize the price vector with zeros for each item, or -infinity for non-items
        self.p = [0 if i in self.item else float('-inf') for i in range(self.nitems)]
        # Track which items have been allocated
        allocated_items = set()

        # Iterate over each item to allocate it to the agent with the highest valuation
        for j in self.item:
            # Find the agent with the highest valuation for item j
            highest_valuation = float('-inf')
            agent_id = None
            for i in self.agent:
                if self.valuation[i][j] > highest_valuation and j not in allocated_items:
                    highest_valuation = self.valuation[i][j]
                    agent_id = i
            
            # Allocate item j to the agent with the highest valuation
            if agent_id is not None:
                self.x[agent_id].append(j)
                allocated_items.add(j)
                # Optionally set the price of the item based on the highest valuation
                self.p[j] = highest_valuation

        # Return the allocation and price vector
        return self.x, self.p

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

    def not_envy(self, i, j):
        """
        Determine if agent i does not envy agent j.

        This method checks if agent i envies agent j based on their allocations and the prices of the items they have.
        An agent i does not envy agent j if either:
        - Agent i has no items and agent j also has no items, or all items that j has result in zero price when removed.
        - Agent i's total valuation (scaled by (1 + ε)) is greater than or equal to agent j's total valuation.
        - Agent i's total valuation (scaled by (1 + ε)) is greater than or equal to agent j's total valuation minus any single item g that j has.

        Inputs:
        - i: The index of agent i.
        - j: The index of agent j.

        Returns:
        - bool: True if agent i does not envy agent j, False otherwise.
        """
        if not self.x[i]:
            # Agent i has no allocation
            if not self.x[j]:
                # Agent j also has no allocation, hence no envy
                return True
            else:
                # Agent j has an allocation
                for g in self.x[j]:
                    price_j_without_g = sum(self.p[item] for item in self.x[j] if item != g)
                    if price_j_without_g == 0:
                        # Price of agent j without item g is zero, hence no envy
                        return True
                # Agent i has no allocation but agent j's allocations have non-zero prices
                return False
        else:
            # Calculate the total price of items allocated to agent i
            price_i = sum(self.p[item] for item in self.x[i])
            # Calculate the total price of items allocated to agent j
            price_j = sum(self.p[item] for item in self.x[j])

            # Check if (1 + ε) times price_i is greater than or equal to price_j
            if (1 + self.epsilon) * price_i >= price_j:
                return True
            else:
                # Check each item in agent j's allocation
                for g in self.x[j]:
                    price_j_without_g = sum(self.p[item] for item in self.x[j] if item != g)
                    if (1 + self.epsilon) * price_i >= price_j_without_g:
                        return True
                # Agent i envies agent j
                return False

        

    def is_3epsilon_pEF1(self):
        """
        Check if the current allocation is 3*epsilon-pEF1.

        This method iterates through each pair of agents to determine if any agent i envies any other agent j.
        If no such envy exists for all pairs, the allocation is considered 3*epsilon-pEF1.

        Returns:
        - bool: True if the allocation is 3*epsilon-pEF1, False otherwise.
        """
        # Iterate through each pair of agents
        for i in self.agent:
            for j in self.agent:
                if i != j:
                    # Check if agent i envies agent j
                    if not self.not_envy(i, j):
                        return False
        # If the condition is met for all pairs, the allocation is 3*epsilon-pEF1
        return True


    def compute_mbb_set(self, agent):
        """
        Compute the maximum bang-per-buck (MBB) set for a given agent.

        This method calculates the bang-per-buck ratio for each item for the specified agent.
        The MBB set consists of items that have the maximum bang-per-buck ratio for that agent.

        Inputs:
        - agent: The index of the agent for whom the MBB set is being calculated.

        Returns:
        - mbb_set: A list of item indices that have the maximum bang-per-buck ratio for the given agent.
        - max_bang_per_buck: The maximum bang-per-buck ratio for the given agent.
        """
        # Initialize bang per buck ratios with 0 for items and -inf for non-items
        bang_per_buck = [0 if i in self.item else float('-inf') for i in range(self.nitems)]
        
        # Iterate over each item to compute the bang per buck ratio
        for item in self.item:
            # Ensure we don't divide by zero
            if self.p[item] != 0:
                bang_per_buck[item] = self.valuation[agent][item] / self.p[item]
            else:
                bang_per_buck[item] = 0
        
        # Find the maximum bang per buck ratio
        max_bang_per_buck = np.max(list(bang_per_buck))

        # List to hold the items with max bang per buck for the agent
        mbb_set = np.where(bang_per_buck == max_bang_per_buck)[0].tolist()

        return mbb_set, max_bang_per_buck


    def find_connected_agents(self, level):
        """
        Find agents connected to a given hierarchy level.

        This method identifies agents connected to the specified hierarchy level by computing the MBB sets
        of agents in the current level and determining which other agents own items in these sets.

        Inputs:
        - level: The hierarchy level for which to find connected agents.

        Returns:
        - connected_agents: A list of agents that are connected to the specified hierarchy level.
        """
        # Get the agents at the specified hierarchy level
        current_level_agents = self.hierarchy.get(level, [])
        # Initialize a set to hold the connected agents
        connected_agents = set()
        # Get all agents that are part of the hierarchy
        hierarchy_agents = self.agent_in_hierarchy()
        # List agents not in the hierarchy
        agents_not_in_hierarchy = [agent for agent in self.agent if agent not in hierarchy_agents]

        # Iterate over each agent in the current level
        for current_agent in current_level_agents:
            # Compute the MBB set for the current agent
            mbb_set, _ = self.compute_mbb_set(current_agent)
            # List to hold the owners of goods in the MBB set
            owners = []
            for good in mbb_set:
                for agent in agents_not_in_hierarchy:
                    if agent != current_agent:
                        if good in self.x[agent]:
                            owners.append(agent)
                            break

            # Add the owners to the connected agents set
            if owners:
                for owner in owners:
                    for h in self.hierarchy.values():
                        if owner not in h and owner not in connected_agents:
                            connected_agents.add(owner)

        return list(connected_agents)

    def build_hierarchy(self, source_agent):
        """
        Build a hierarchy of agents starting from a source agent.

        This method constructs a hierarchical structure of agents, where each level contains agents
        connected to agents in the previous level. The hierarchy is built starting from the specified
        source agent.

        Inputs:
        - source_agent: The index of the agent to start the hierarchy from.

        Returns:
        - None
        """
        self.hierarchy = {0: [source_agent]}  # Initialize hierarchy with source agent at level 0
        l = 0  # Starting level

        # Loop to build hierarchy level by level
        while True:
            # Find agents connected to the current level
            next_level_agents = self.find_connected_agents(l)

            # Check if there are agents to add to the next level and the level is within bounds
            if next_level_agents and l < len(self.agent) - 1:
                l += 1
                self.hierarchy[l] = next_level_agents
            else:
                break  # Stop if no agents are connected to the current level or maximum level is reached


    def build_alternating_path(self, source_agent, target_agent, level):
        """
        Build an alternating path from the source agent to the target agent.

        This method constructs an alternating path in the allocation graph starting from the source agent and
        ending at the target agent. The path alternates between agents and goods allocated to them.

        Inputs:
        - source_agent: The index of the agent to start the path from.
        - target_agent: The index of the agent to end the path at.
        - level: The current level in the hierarchy of agents.

        Returns:
        - path: A list representing the alternating path from the source agent to the target agent.
                If no path is found, returns None.
        """
        # Initialize the path with the target agent
        path = [target_agent]
        current_agent = target_agent
        l = level

        while current_agent != source_agent:
            found_next_agent = False
            # Iterate over each good allocated to the current agent
            for good in self.x[current_agent]:
                next_agents = []
                # Check if the good is allocated to an agent closer to the target in the hierarchy
                for i in range(l - 1, -1, -1):
                    for agent in self.hierarchy[i]:
                        mbb_set, _ = self.compute_mbb_set(agent)
                        if good in mbb_set:
                            next_agents = [agent]
                            break
                    if next_agents:
                        break

                if next_agents:
                    # Move to the next agent in the path
                    next_agent = next_agents[0]  # Assuming there's at least one such agent
                    path.append(good)
                    path.append(next_agent)
                    current_agent = next_agent
                    found_next_agent = True
                    break

            if not found_next_agent:
                # If no next agent is found, return None indicating no path
                return None

        # Return the constructed path if it ends at the source agent, otherwise return None
        return path if path[-1] == source_agent else None

        
    def e_path_violator(self, level, least_spender, price_least_spender):
        """
        Determine if agent k is an ε-path violator along the alternating path P in the hierarchy.
        
        Parameters:
        h (int): Index of the current agent being considered.
        hierarchy (list of int): The hierarchy of agents.
        k (int): Position of agent h in the hierarchy.
        
        Returns:
        bool: True if agent h is an ε-path violator, False otherwise.
        """
        # Assuming self.x is the allocation matrix where self.x[h] is the bundle of agent h
        # and self.p is the price vector where self.p[j] is the price of item j
        
        # Check if there is any item in agent h's bundle such that if removed,
        # h would prefer their bundle without that item, given the price of the item is increased by ε%
        h = self.hierarchy[level]
        for agent in h:
            alternating_path = self.build_alternating_path(least_spender, agent, level)
            # print("Alternating path: ", alternating_path)
            if alternating_path:
                good = alternating_path[1]
                # Check if removing any item from agent agent's bundle reduces the bundle price wrt least spender agent bundle price
                price_agent_without_good = sum(self.p[item] for item in self.x[agent] if item != good)
                if price_agent_without_good > (1 + self.epsilon) * price_least_spender:
                    return True, alternating_path
                
        return False, None
             

    def perform_swap(self, alternation_path):
        """
        Perform a swap of items between agents based on the given alternation path.

        This method swaps items between the current ε-path violator and other agents in the hierarchy
        according to the alternation path.

        Inputs:
        - alternation_path: A list representing the alternating path, where agents and goods alternate.

        Returns:
        - None
        """
        # 'h' is the current ε-path violator, and 'k' is their position in the hierarchy.
        # Determine which items to swap between 'h' and other agents in the hierarchy.

        # For the violator 'h' and each agent above 'h' in the hierarchy:
        last_agent = alternation_path[0]
        good = alternation_path[1]
        previous_agent = alternation_path[2]

        # Swap the good between the violator and the previous agent
        self.x[last_agent].remove(good)
        self.x[previous_agent].append(good)


    def phase_1_initialization(self):
        """
        Perform Phase 1 Initialization for the allocation process.

        This method initializes the allocation by first computing the welfare-maximizing allocation
        and then initializes prices based on the current allocation and other criteria.

        Returns:
        - None
        """
        # Compute the welfare-maximizing allocation
        self.welfare_maximizing_allocation()
        # Initialize prices p based on x and other criteria (details not provided)
        # Further implementation needed here for initializing prices based on the specific criteria


    def phase_2_and_phase_3(self):
        """
        Perform Phase 2 and Phase 3 of the allocation process.

        This method iteratively checks and adjusts the allocation to ensure it meets the 3*epsilon-pEF1 condition.
        It involves identifying the least spender, building a hierarchy, and potentially swapping items between agents.

        Returns:
        - tuple: A tuple containing the allocation (self.x) and the price vector (self.p).
        """
        while not self.is_3epsilon_pEF1():
            # Identify the least spender and build the hierarchy starting from them
            i = self.least_spender()
            self.build_hierarchy(i)
            k = 1  # Initialize k to 1
            price_least_spender = sum(self.p[item] for item in self.x[i])

            while k in self.hierarchy and not self.is_3epsilon_pEF1():
                level = k
                # Check if there is an ε-path violator at the current level
                there_is_e_path_violator, alternating_path = self.e_path_violator(level, i, price_least_spender)

                if there_is_e_path_violator:
                    # Perform swap operation using the alternating path
                    self.perform_swap(alternating_path)
                    # Recur to check if the swap resolved the envy
                    self.phase_2_and_phase_3()
                else:
                    # Move to the next agent in the hierarchy
                    k += 1

            if self.is_3epsilon_pEF1():
                return self.x, self.p
            else:
                # Move to Phase 3 if the allocation is not 3ε-pEF1
                x_h = self.element_in_hierarchy()
                a_h = self.agent_in_hierarchy()

                # Compute alpha1, alpha2, and alpha3 based on the pseudocode
                alpha1 = self.raising_prices_alpha1(i, x_h, a_h)
                alpha2 = self.raising_prices_alpha2(i, a_h)
                alpha3 = self.raising_prices_alpha3(i, a_h)
                # Determine the smallest alpha
                alpha = min(alpha1, alpha2, alpha3)

                # Adjust the prices for items in x_h
                for j in x_h:
                    self.p[j] *= alpha

                if alpha == alpha2:
                    return self.x, self.p

                

        
    
    def raising_prices_alpha1(self, i, x_h, a_h):
        """
        Compute the raising price factor alpha1 for Phase 3.

        This method calculates the minimum alpha1 value needed to raise prices in Phase 3,
        ensuring that the maximum bang-per-buck ratio is maintained for agents in a_h and items in x_h.

        Inputs:
        - i: The index of the agent to start the path from.
        - x_h: A set of items in the hierarchy.
        - a_h: A set of agents in the hierarchy.

        Returns:
        - min_alpha1: The minimum alpha1 value required to adjust prices.
        """
        min_alpha1 = float('inf')  # Initialize min_alpha1 to infinity

        for agent in a_h:
            _, ratio_agent = self.compute_mbb_set(agent)
            if ratio_agent == 0:
                min_alpha1 = 0
                break

            for good in self.item:
                if good not in x_h:
                    if self.p[good] > 0 and self.valuation[agent][good] > 0:
                        alpha1 = ratio_agent / (self.valuation[agent][good] / self.p[good])
                        if alpha1 < min_alpha1:
                            min_alpha1 = alpha1

                    elif self.p[good] == 0 and self.valuation[agent][good] > 0:
                        alpha1 = 0
                        if alpha1 < min_alpha1:
                            min_alpha1 = alpha1

                    elif self.valuation[agent][good] == 0:
                        alpha1 = float('inf')
                        if alpha1 < min_alpha1:
                            min_alpha1 = alpha1

        return min_alpha1


    def raising_prices_alpha2(self, i, a_h):
        """
        Compute the raising price factor alpha2 for Phase 3.

        This method calculates the alpha2 value needed to raise prices in Phase 3,
        ensuring that the least spender's total price is proportionally adjusted based on other agents' minimum prices.

        Inputs:
        - i: The index of the least spender agent.
        - a_h: A set of agents in the hierarchy.

        Returns:
        - alpha2: The alpha2 value required to adjust prices.
        """
        # Calculate the total price of items allocated to the least spender
        price_least_spender = sum(self.p[item] for item in self.x[i])

        max_alpha2 = float('-inf')  # Initialize max_alpha2 to negative infinity

        for agent in self.agent:
            if agent not in a_h:
                min_price = float('inf')  # Initialize min_price to infinity
                for good in self.x[agent]:
                    # Calculate the price of the agent's allocation excluding the current good
                    price_j = sum(self.p[item] for item in self.x[agent]) - self.p[good]
                    if price_j < min_price:
                        min_price = price_j
                if min_price > max_alpha2:
                    max_alpha2 = min_price

        if price_least_spender == 0:
            alpha2 = float('inf')
        else:
            alpha2 = max_alpha2 / price_least_spender

        return alpha2


    def raising_prices_alpha3(self, i, a_h):
        """
        Compute the raising price factor alpha3 for Phase 3.

        This method calculates the alpha3 value needed to raise prices in Phase 3,
        ensuring that the least spender's total price is proportionally adjusted based on the minimum price outside the hierarchy.

        Inputs:
        - i: The index of the least spender agent.
        - a_h: A set of agents in the hierarchy.

        Returns:
        - alpha3: The alpha3 value required to adjust prices.
        """
        min_price_outside_h = float('inf')  # Initialize min_price_outside_h to infinity

        for agent in self.agent:
            if agent not in a_h:
                # Calculate the total price of items allocated to the agent outside the hierarchy
                price_outside_h = sum(self.p[item] for item in self.x[agent])
                if price_outside_h < min_price_outside_h:
                    min_price_outside_h = price_outside_h

        # Calculate the total price of items allocated to the least spender
        least_spender_price = sum(self.p[item] for item in self.x[i])
        if least_spender_price == 0:
            s = float('inf')
        else:
            # Calculate the ratio of the minimum price outside the hierarchy to the least spender's price
            ratio = min_price_outside_h / least_spender_price
            if ratio == 1:
                s = 1
            else:
                s = np.ceil(np.log(min_price_outside_h / least_spender_price) / np.log(1 + self.epsilon))

        # Compute alpha3 based on the calculated s value
        alpha3 = (1 + self.epsilon) ** s

        return alpha3
        

    def element_in_hierarchy(self):
        """
        Collect all items allocated to agents in the hierarchy.

        This method iterates through each level of the hierarchy and aggregates all items allocated
        to agents in these levels.

        Returns:
        - x_h: A list of items that are allocated to agents in the hierarchy.
        """
        x_h = []  # Initialize an empty list to hold items in the hierarchy

        for level in self.hierarchy:
            for agent in self.hierarchy[level]:
                x_h = x_h + self.x[agent]  # Append items allocated to the current agent

        return x_h

    
    def agent_in_hierarchy(self):
        """
        Collect all agents in the hierarchy.

        This method iterates through each level of the hierarchy and aggregates all agents
        in these levels into a single list.

        Returns:
        - a_h: A list of agents that are in the hierarchy.
        """
        a_h = []  # Initialize an empty list to hold agents in the hierarchy

        for level in self.hierarchy:
            for agent in self.hierarchy[level]:
                a_h.append(agent)  # Append the current agent to the list

        return a_h


    def run_algorithm(self):
        """
        Execute the main algorithm for fair allocation.

        This method performs the following steps:
        1. Rounds valuations to be multiples of epsilon.
        2. Initializes the allocation in Phase 1.
        3. Checks if the allocation meets the 3*epsilon-pEF1 condition.
        4. If not, proceeds to Phase 2 and Phase 3 to adjust the allocation.

        Returns:
        - tuple: A tuple containing the final allocation (self.x) and the price vector (self.p).
        """
        # Round valuations to multiples of epsilon
        self.epsilon_round_valuations()  
        self.phase_1_initialization()  # Perform Phase 1 initialization

        if not self.is_3epsilon_pEF1():
            self.phase_2_and_phase_3()  # Proceed to Phase 2 and Phase 3 if needed

        return self.x, self.p  # Return the final allocation and price vector

    def least_spender(self):
        """
        Identify the agent with the least total spending.

        This method calculates the total spending of each agent and returns the one with the minimum spending.

        Returns:
        - least_spender_agent: The index of the agent with the least spending.
        """
        min_spending = float('inf')  # Initialize minimum spending to infinity
        least_spender_agent = None  # Initialize the least spender agent to None

        for i in self.agent:
            # Calculate the total spending of agent i
            agent_spending = sum(self.p[item] for item in self.x[i])

            # Update min_spending and least_spender_agent if this agent's spending is lower
            if agent_spending < min_spending:
                min_spending = agent_spending
                least_spender_agent = i

        return least_spender_agent  # Return the agent with the least spending

