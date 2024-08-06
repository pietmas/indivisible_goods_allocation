import numpy as np
 
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
        v_max = np.float64(max(max(row) for row in self.valuation))
        epsilon = 0
        # Number of items
        m = self.nitems
        # Check if the maximum valuation is zero
        if v_max == 0:
            # Raise an error if the valuation matrix contains only zeros
            raise ValueError("Valuation matrix contains only zeros")
        else:
            # Calculate epsilon using the formula ε = 1 / (14 * m^3 * v_max^4)
            try:
                log_epsilon = -np.log(14) - 3 * np.log(m) - 4 * np.log(v_max)
                epsilon = np.exp(log_epsilon)
            except OverflowError:
                epsilon = 0  # Handle overflow case
        return epsilon
        
    def epsilon_round_valuations(self):
        """
        Adjusts each valuation in the matrix to be a power of 1 + epsilon.

        This method rounds each non-zero valuation in the valuation matrix to the nearest power of 1 + ε,
        ensuring the adjusted valuations are more manageable for the algorithm.

        Returns:
        None
        """
        # Iterate over each agent's valuations
        for i in range(self.nagents):
            # Iterate over each item for the current agent
            for j in range(self.nitems):
                if self.epsilon > 0 and self.valuation[i][j] > 0:  # Ensure ε is positive to avoid division by zero
                    # Round each valuation to the nearest multiple of ε
                    self.valuation[i][j] = (1 + self.epsilon) ** np.ceil(np.log(self.valuation[i][j]) / np.log(1 + self.epsilon))
                elif self.valuation[i][j] == 0:
                    # If the valuation is zero, it remains zero
                    self.valuation[i][j] = 0
        
    def welfare_maximizing_allocation(self):
        """
        Finds a welfare-maximizing allocation of items to agents based on their valuations.

        This method allocates each item to the agent with the highest valuation for that item, ensuring no item is allocated more than once.

        Returns:
        tuple: A tuple containing the allocation matrix and the price vector.
               - allocation matrix (numpy.ndarray): A matrix where each row represents an agent's allocation.
               - price vector (list): A list of prices for each item.
        """
        # Initialize the allocation matrix as a list of lists, where each sublist represents an agent's allocation
        self.x = np.zeros((self.nagents, self.nitems), dtype=int)

        # Initialize the price vector with zeros for each item
        self.p = [0] * self.nitems
        # Track which items have been allocated

        # Iterate over each item to allocate it to the agent with the highest valuation
        for j in self.item:
            # Find the agent with the highest valuation for item j
            agent_id = np.argmax(self.valuation[:, j])

            # Allocate item j to the agent with the highest valuation
            if agent_id is not None:
                self.x[agent_id][j] = 1
                # Optionally set initial prices, e.g., based on valuation
                self.p[j] = self.valuation[agent_id][j]

        # Return the allocation matrix and price vector
        return self.x, self.p

    def not_envy(self, i, j):
        """
        Checks if agent i does not price envy agent j.

        This method evaluates whether agent i does not epsilon-price-envy agent j by comparing the total price of items allocated
        to both agents and considering the epsilon value for approximation.

        Parameters:
        i (int): The index of agent i.
        j (int): The index of agent j.

        Returns:
        bool: True if agent i does not price envy agent j, False otherwise.
        """
        # Check if agent i has no allocated items
        if not self.x[i].any():
            # Check if agent j also has no allocated items
            if not self.x[j].any():
                return True
            else:
                # Iterate through items allocated to agent j
                for g in np.where(self.x[j] == 1)[0]:
                    price_j = np.dot(self.p, self.x[j])
                    price_j_without_g = price_j - self.p[g]
                    # Check if the price without item g is zero
                    if price_j_without_g == 0:
                        return True
                return False
        else:
            price_i = np.dot(self.p, self.x[i])
            price_j = np.dot(self.p, self.x[j])
            # Check if agent i's price is greater than or equal to agent j's price by an epsilon factor
            if (1 + self.epsilon) * price_i >= price_j:
                return True
            else:
                # Iterate through items allocated to agent j
                for g in np.where(self.x[j] == 1)[0]:
                    price_j_without_g = price_j - self.p[g]
                    # Check if agent i's price is greater than or equal to agent j's price without item g by an epsilon factor
                    if (1 + self.epsilon) * price_i >= price_j_without_g:
                        return True
                return False
        

    def is_3epsilon_pEF1(self, agent=None):
        """
        Checks if the current allocation is 3*epsilon-pEF1 (price Envy-Free up to one item).

        This method iterates through each pair of agents to check if the allocation meets the 3*epsilon-pEF1 condition,
        which means no agent envies another agent's allocation considering the epsilon value.

        Parameters:
        agent (list of int, optional): A list of agent indices to check. If None, all agents are checked.

        Returns:
        bool: True if the allocation is 3*epsilon-pEF1 for the specified agents, False otherwise.
        """
        # Use all agents if no specific agent list is provided
        if agent is None:
            agent = self.agent

        # Iterate through each pair of agents
        for i in agent:
            for j in agent:
                if i != j:
                    # Check if agent i envies agent j
                    if not self.not_envy(i, j):
                        return False

        # If the condition is met for all pairs, the allocation is 3*epsilon-pEF1
        return True


    def compute_mbb_set(self, agent):
        """
        Compute the Maximum Bang per Buck (MBB) set for a given agent.
        
        This method calculates the MBB set, which consists of goods that provide the highest value per unit cost for the specified agent.

        Parameters:
        agent (int): The index of the agent for whom to calculate the MBB set.

        Returns:
        tuple: A tuple containing:
            - list: A list of items that maximize the bang per buck ratio for the agent.
            - float: The maximum bang per buck ratio.
        """
        # Initialize the bang per buck ratio for each item
        bang_per_buck = [0 if i in self.item else -np.inf for i in range(self.nitems)]
        
        # Iterate over each item to compute the bang per buck for the given agent
        for item in range(self.nitems):
            # Ensure we don't divide by zero
            if self.p[item] != 0:
                bang_per_buck[item] = self.valuation[agent][item] / self.p[item]
            else:
                bang_per_buck[item] = 0
        
        # Find the maximum bang per buck ratio
        max_bang_per_buck = np.max(bang_per_buck)

        # List to hold the items with max bang per buck for the agent
        mbb_set = np.where(bang_per_buck == max_bang_per_buck)[0].tolist()

        return mbb_set, max_bang_per_buck


    def find_connected_agents(self, level):
        """
        Finds agents connected via MBB-allocation edge at a given hierarchy level.

        This method identifies agents that are connected to the current level agents through the Maximum Bang per Buck (MBB) allocation.

        Parameters:
        level (int): The hierarchy level to process.

        Returns:
        list: A list of connected agents not yet in the hierarchy.
        """
        # Get the list of agents at the current hierarchy level
        current_level_agents = self.hierarchy.get(level, [])
        connected_agents = set()
        # Get the list of agents already in the hierarchy
        hierarchy_agent = self.agent_in_hierarchy()
        # Get the list of agents not in the hierarchy
        agents_not_in_hierarchy = [agent for agent in self.agent if agent not in hierarchy_agent]

        # Iterate over each agent at the current hierarchy level
        for current_agent in current_level_agents:
            # Compute the MBB set for the current agent
            mbb_set, _ = self.compute_mbb_set(current_agent)
            owners = []
            # Iterate over each good in the MBB set
            for good in mbb_set:
                # Check which agents own the goods in the MBB set
                for agent in agents_not_in_hierarchy:
                    # Iterate over each good allocated to the agent
                    if self.x[agent][good] == 1:
                        owners.append(agent)
                        break

                # Add the owners to the connected agents if they are not in the current hierarchy level
                if owners:
                    for owner in owners:
                        for h in self.hierarchy.values():
                            if owner not in h and owner not in connected_agents:
                                connected_agents.add(owner)

        return list(connected_agents)


    def build_hierarchy(self, source_agent):
        """
        Builds a hierarchy of agents starting from the source agent.

        This method constructs the hierarchy level by level, starting from the source agent at level 0 and adding connected agents at subsequent levels.

        Parameters:
        source_agent (int): The index of the source agent to start building the hierarchy from.

        Returns:
        None
        """
        self.hierarchy = {0: [source_agent]}  # Initialize with source agent at level 0
        l = 0  # Starting level

        # Loop to build hierarchy level by level
        while True:
            # Find agents connected to the current level agents
            next_level_agents = self.find_connected_agents(l)
            # If there are agents to add to the next level and the level is within the total number of agents
            if next_level_agents and l < len(self.agent) - 1:
                l += 1
                self.hierarchy[l] = next_level_agents
            else:
                break  # Stop if no agents are connected to the current level

    def build_alternating_path(self, source_agent, target_agent, level):
        """
        Builds an alternating path from the source agent to the target agent.

        This method constructs a path starting from the target agent, alternating between goods and agents, moving up through the hierarchy
        until it reaches the source agent or determines that no such path exists.

        Parameters:
        source_agent (int): The index of the source agent.
        target_agent (int): The index of the target agent.
        level (int): The current hierarchy level of the target agent.

        Returns:
        list: A list representing the alternating path of agents and goods, or None if no such path exists.
        """
        # Initialize the path with the target agent
        path = [target_agent]
        current_agent = target_agent

        while current_agent != source_agent:
            found_next_agent = False

            # Iterate over each good allocated to the current agent
            for good in np.where(self.x[current_agent] == 1)[0]:
                # Check which agents own the goods in the MBB set
                for agent in self.hierarchy[level - 1]:
                    
                    # Compute the MBB set for the agent
                    mbb_set, _ = self.compute_mbb_set(agent)
                    # Check if the good is in the MBB set
                    if good in mbb_set:
                        level -= 1
                        # Add the good to the path
                        path.append(good)
                        # Add the agent to the path
                        path.append(agent)
                        current_agent = agent
                        found_next_agent = True
                        break
                if found_next_agent:
                    break
            if not found_next_agent:
                return None
            
        return path

    def least_spender(self):
        """
        Identifies the agent with the least spending.

        It then returns the agent with the minimum spending.

        Returns:
        int: The index of the agent with the least spending.
        """
        # Initialize a variable to keep track of the minimum spending and the corresponding agent
        min_spending = np.inf  # Start with infinity to ensure any real spending is lower
        least_spender_agent = None

        # Iterate through each agent to calculate their total spending
        for i in self.agent:
            # Calculate the total spending of agent i
            agent_spending = np.dot(self.p, self.x[i])

            # Update min_spending and least_spender_agent if this agent's spending is lower
            if agent_spending < min_spending:
                min_spending = agent_spending
                least_spender_agent = i

        return least_spender_agent

        
    def e_path_violator(self, level, least_spender, price_least_spender):
        """
        Determine if an agent at a given hierarchy level is an ε-path violator along the alternating path P in the hierarchy.

        Parameters:
        level (int): The hierarchy level being considered.
        least_spender (int): The index of the agent who is the least spender.
        price_least_spender (float): The price of the least spender's bundle.

        Returns:
        tuple: A tuple containing:
            - bool: True if an agent at the given hierarchy level is an ε-path violator, False otherwise.
            - list: The alternating path if an ε-path violator is found, None otherwise.
        """
        # Get the list of agents at the current hierarchy level
        h = self.hierarchy[level]
        
        # Iterate over each agent at the current hierarchy level
        for agent in h:
            # Build the alternating path from the least spender to the current agent
            alternating_path = self.build_alternating_path(least_spender, agent, level)

            
            if alternating_path:
                # The good in the alternating path to check
                good = alternating_path[1]
                
                # Calculate the price of the agent's bundle without the specific good
                price_agent_without_good = sum(self.p[item] for item in np.where(self.x[agent] == 1)[0] if item != good)
                
                # Check if removing the item from the agent's bundle results in a price greater than (1 + ε) times the least spender's bundle price
                if price_agent_without_good > (1 + self.epsilon) * price_least_spender:
                    return True, alternating_path

        return False, None
             

    def perform_swap(self, alternation_path):
        """
        Performs a swap of goods between agents along the provided alternating path.

        This method swaps the specified good between the ε-path violator and the previous agent in the hierarchy.

        Parameters:
        alternation_path (list): A list representing the alternating path of agents and goods.

        Returns:
        None
        """
        # 'last_agent' is the current ε-path violator, and 'previous_agent' is the agent to swap with.
        last_agent = alternation_path[0]
        good = alternation_path[1]
        previous_agent = alternation_path[2]
        
        # Swap the good between the violator and the previous agent
        self.x[last_agent][good] = 0
        self.x[previous_agent][good] = 1


    def phase_1_initialization(self):
        """
        Performs phase 1 initialization by computing the welfare-maximizing allocation
        and initializing prices based on the allocation and other criteria.

        This method first computes the welfare-maximizing allocation and then initializes the prices
        based on the resulting allocation and any additional criteria.

        Returns:
        None
        """
        # Compute the welfare-maximizing allocation
        self.welfare_maximizing_allocation()
        

    def phase_2_and_phase_3(self):
        """
        Executes phases 2 and 3 of the allocation algorithm, iteratively ensuring the allocation
        satisfies the 3ε-pEF1 condition by identifying and resolving ε-path violators.

        The method identifies the least spender, builds a hierarchy of agents, checks for ε-path violators,
        performs necessary swaps, and adjusts prices to achieve the desired allocation properties.

        Returns:
        tuple: A tuple containing the final allocation matrix and price vector.
        """
        while not self.is_3epsilon_pEF1():
            # Identify the least spender and build the hierarchy starting from them
            i = self.least_spender()
            self.build_hierarchy(i)
            k = 1  # Initialize k to 1
            # Calculate the price of the least spender's bundle
            price_least_spender = np.dot(self.p, self.x[i])
            # print(f"Least spender: {i}, price: {price_least_spender}\n")

            while k in self.hierarchy and not self.is_3epsilon_pEF1():
                level = k
                # print(f"Level: {level}\n")

                # Check for ε-path violators at the current hierarchy level
                there_is_e_path_violator, alternating_path = self.e_path_violator(level, i, price_least_spender)
                # print(f"Path violator: {there_is_e_path_violator}\n")
                # print(f"Alternating path: {alternating_path}\n")
                if there_is_e_path_violator:
                    # print(f"Performing swap\n")

                    # Perform the swap operation for the identified ε-path violator
                    self.perform_swap(alternating_path)

                    # Recur to check if the swap resolved the envy
                    self.phase_2_and_phase_3()
                else:
                    k += 1  # Move to the next agent in the hierarchy

            if self.is_3epsilon_pEF1():
                return self.x, self.p
            else:
                # Move to Phase 3 if not 3ε-pEF1
                x_h = self.elements_in_hierarchy()
                a_h = self.agent_in_hierarchy()

                # Compute alpha1, alpha2, and alpha3 based on the pseudocode
                alpha1 = self.raising_prices_alpha1(i, x_h, a_h)
                alpha2 = self.raising_prices_alpha2(i, a_h)
                alpha3, alpha3_2 = self.raising_prices_alpha3(i, a_h)

                if alpha3_2:
                    alpha3 = alpha3_2
                    
                # Determine the smallest alpha
                alpha = min(alpha1, alpha2, alpha3)

                # Adjust the prices
                for j in x_h:
                    self.p[j] *= alpha

                if alpha == alpha2:
                    return self.x, self.p

            
    def raising_prices_alpha1(self, i, x_h, a_h):
        """
        Computes the alpha1 value for raising prices during the allocation process.

        This method calculates alpha1, which determines how much to raise prices until a new agent gets added to the hierarchy.

        Parameters:
        i (int): The index of the least spender agent.
        x_h (list of int): The list of items in the hierarchy.
        a_h (list of int): The list of agents in the hierarchy.

        Returns:
        float: The minimum alpha1 value.
        """
        min_alpha1 = np.inf

        for agent in a_h:
            # Compute the MBB set and ratio for the current agent
            _, ratio_agent = self.compute_mbb_set(agent)
            for good in self.item:
                if good not in x_h:
                    if self.p[good] > 0 and self.valuation[agent][good] > 0:
                        # Calculate alpha1 for goods with positive prices and positive valuations
                        alpha1 = ratio_agent / (self.valuation[agent][good] / self.p[good])
                        if alpha1 < min_alpha1:
                            min_alpha1 = alpha1
                    elif self.p[good] == 0 and self.valuation[agent][good] > 0:
                        # If the price of the good is zero and the valuation is positive, set alpha1 to 0
                        alpha1 = 0
                        if alpha1 < min_alpha1:
                            min_alpha1 = alpha1
                    elif self.valuation[agent][good] == 0:
                        # If the valuation of the good is zero, set alpha1 to infinity
                        alpha1 = np.inf
                        if alpha1 < min_alpha1:
                            min_alpha1 = alpha1

        return min_alpha1


    def raising_prices_alpha2(self, i, a_h):
        """
        Computes the alpha2 value for raising prices during the allocation process.

        This method calculates alpha2, which determines how much to raise prices until the pEF1 condition is satisfied.

        Parameters:
        i (int): The index of the least spender agent.
        a_h (list of int): The list of agents in the hierarchy.

        Returns:
        float: The computed alpha2 value.
        """
        # Calculate the price of the least spender's bundle
        price_least_spender = np.dot(self.p, self.x[i])

        max_alpha2 = -np.inf

        for agent in self.agent:
            if agent not in a_h:
                min_price = np.inf
                # Iterate over each good allocated to the agent
                for good in np.where(self.x[agent] == 1)[0]:
                    # Calculate the price of the agent's bundle without the specific good
                    price_j_without_g = np.dot(self.p, self.x[agent]) - self.p[good]
                    if price_j_without_g < min_price:
                        min_price = price_j_without_g
                if min_price > max_alpha2:
                    max_alpha2 = min_price

        if price_least_spender == 0:
            alpha2 = np.inf
        else:
            alpha2 = max_alpha2 / price_least_spender

        return alpha2
 

    def raising_prices_alpha3(self, i, a_h):
        """
        Computes the alpha3 value for raising prices during the allocation process.

        This method calculates alpha3, which determines how much to raise prices to maintain the 3ε-pEF1 condition.
        It finds the minimum price of agents outside the hierarchy and compares it with the least spender's bundle price.

        Parameters:
        i (int): The index of the least spender agent.
        a_h (list of int): The list of agents in the hierarchy.

        Returns:
        float: The computed alpha3 value.
        """
        min_price_outside_h = np.inf
        # List to store prices
        prices = []

        for agent in self.agent:
            if agent not in a_h:
                price_agent = np.dot(self.p, self.x[agent])
                prices.append(price_agent)
        
        least_spender_price = np.dot(self.p, self.x[i])

        # Ensure there are at least two prices to compare
        if len(prices) < 2:
            if least_spender_price == 0:
                return np.inf, None
            else:
                s_1 = np.ceil(np.log(min_price_outside_h / least_spender_price) / np.log(1 + self.epsilon))
                
                alpha3 = (1 + self.epsilon) ** s_1
                return alpha3, None
        else:
            # Sort the prices in ascending order
            sorted_prices = sorted(prices)

            # Extract the two minimum prices
            min_price = sorted_prices[0]
            second_min_different_price = 0
            for price in sorted_prices:
                if price != min_price:
                    second_min_different_price = price
                    break
            second_min_price = second_min_different_price

            # Calculate alpha2 based on the minimum prices
            if least_spender_price == 0:
                alpha3 = np.inf
                alpha3_2 = np.inf
            else:
                ratio_1 = min_price / least_spender_price
                ratio_2 = second_min_price / least_spender_price
                if ratio_1 == 1:
                    s_1 = 1
                else:
                    s_1 = np.ceil(np.log(min_price / least_spender_price) / np.log(1 + self.epsilon))
                if ratio_2 == 1:
                    s_2 = 1
                else:
                    s_2 = np.ceil(np.log(second_min_price / least_spender_price) / np.log(1 + self.epsilon))
                
                alpha3 = (1 + self.epsilon) ** s_1
                alpha3_2 = (1 + self.epsilon) ** s_2

            return alpha3, alpha3_2

    def elements_in_hierarchy(self):
        """
        Collects all the elements (goods) in the hierarchy.

        This method iterates through the hierarchy and gathers all the goods allocated to the agents at each level.

        Returns:
        list: A list of all goods present in the hierarchy.
        """
        x_h = []
        # Iterate through each level in the hierarchy
        for level in self.hierarchy:
            # Iterate through each agent at the current hierarchy level
            for agent in self.hierarchy[level]:
                # Collect all goods allocated to the agent and add them to x_h
                x_h += np.where(self.x[agent] == 1)[0].tolist()
        return x_h

    
    def agent_in_hierarchy(self):
        """
        Collects all the agents in the hierarchy.

        This method iterates through the hierarchy and gathers all the agents at each level.

        Returns:
        list: A list of all agents present in the hierarchy.
        """
        a_h = []
        # Iterate through each level in the hierarchy
        for level in self.hierarchy:
            # Iterate through each agent at the current hierarchy level
            for agent in self.hierarchy[level]:
                a_h.append(agent)
        return a_h


    def run_algorithm(self):
        """
        Runs the complete allocation algorithm.

        This method performs the following steps:
        1. Rounds the valuations using epsilon.
        2. Initializes the first phase of the allocation.
        3. Checks if the allocation satisfies the 3ε-pEF1 condition.
        4. If not, it runs phases 2 and 3 to achieve the desired allocation properties.

        Returns:
        tuple: A tuple containing the final allocation matrix and price vector.
        """
        # Round the valuations using epsilon
        self.epsilon_round_valuations()
        # Initialize the first phase of the allocation
        self.phase_1_initialization()

        # Check if the allocation satisfies the 3ε-pEF1 condition
        if not self.is_3epsilon_pEF1():
            # Run phases 2 and 3 if the condition is not met
            self.phase_2_and_phase_3()

        # Return the final allocation matrix and price vector
        return self.x, self.p



