import numpy as np
import numpy as np
from itertools import chain, combinations, permutations


class MinimaxTrade:
    """
    This class implements an algorithm for allocating goods among agents in a way that minimizes envy and maximizes fairness.
    The allocation is performed using Pareto exchanges and checks for envy-freeness properties to ensure a fair distribution of items.

    Attributes:
    - n (int): Number of agents (fixed at 2 in the initializer).
    - m (int): Number of goods.
    - agents (list): List of agents represented by their indices.
    - items (list): List of items represented by their indices.
    - valuation (numpy.ndarray): Valuation matrix where valuation[i][j] represents the valuation of agent i for good j.
    - allocation (numpy.ndarray): 2D array to store the allocation result where allocation[i][j] = 1 if item j is allocated to agent i.

    Methods:
    - envy(allocation): Computes the maximal envy between the agents based on the given allocation.
    - not_ef(allocation): Checks if the given allocation is not Envy-Free (EF) but EF1.
    - minimax_trade(): Implements a Minimax Trade algorithm to allocate items among agents.
    - minimal_envy_bundles(maximal_pareto_exchanges): Determines the minimal envy bundles from the given minimal Pareto exchanges.
    - trade(minimal_pareto_exchange, preferred_agent, preferred_item): Executes the trade based on the minimal Pareto exchange to allocate the preferred item.
    - compute_utility_matrix(allocation_matrix): Computes the utility matrix based on the given allocation matrix.
    - find_pareto_exchanges(allocation_matrix, item): Finds Pareto exchanges for a given allocation matrix and item.
    - find_maximal_pareto_improvements(pareto_exchanges): Finds the maximal Pareto improvements from a list of Pareto exchanges.
    """
    def __init__(self, n, m, valuation):
        
        self.n = 2  # number of agents
        self.m = m  # number of goods
        self.agents = list(range(n))
        self.items = list(range(m))
        self.valuation = valuation  # valuation matrix
        self.allocation = np.zeros((self.n, self.m), dtype=int)  # initial empty allocation


    def envy(self, allocation):
        """
        Compute the maximal envy between the agent
        
        Parameters:
        allocation (np.array): A 2D numpy array where allocation[agent][item] is 1 if the item
                                is allocated to the agent, and 0 otherwise.
        
        Returns:
        int: The maximal envy between the agents.
        int: The agent that envies the most.
        """  
        utility_matrix = np.dot(self.valuation, allocation.T)
        envy_matrix = np.zeros((self.n, self.n))
        for i in self.agents:
            for j in self.agents:
                if i == j:
                    continue
                else:
                    envy_matrix[i][j] = utility_matrix[i][j] - utility_matrix[i][i]
        agents_envy = np.max(envy_matrix, axis=1)

        return np.argsort(agents_envy), agents_envy
    
    def not_ef(self, allocation):
        """
        Check if the given allocation is not EF but EF1.
        """


        # Check if the allocation is EF
        utility_matrix = np.dot(self.valuation, allocation.T)
        flag = None
        for i in self.agents:
            ef1_items = []
            for j in self.agents:
                if i == j:
                    continue  # Skip checking against itself
                if utility_matrix[i][i] < utility_matrix[i][j]:

                    flag = "NotEF1"
                    for item in np.where(allocation[j] == 1)[0]:
                        if utility_matrix[i][i] >= utility_matrix[i][j] - self.valuation[i][item]:
                            ef1_items.append(item)
                            flag = "EF1"
                    if len(ef1_items) == sum(allocation[j]):
                        flag = "EFX"
                        break
            if flag:
                break
        if not flag:
            flag = "EF"
                    
        order, max_envy = self.envy(allocation)
        new_flag = None
        
        if flag == "EFX" or flag == "EF1":
            agent_j = order[1]
            agent_i = order[0]
            for item in np.where(allocation[agent_j] == 1)[0]:
                if utility_matrix[agent_j][agent_i] >= utility_matrix[agent_j][agent_j] - self.valuation[agent_j][item]:
                    new_flag = "EFS1"
                    break
        elif flag == "EF":
            new_flag = "EFS1"


        
        if flag == "EFX":
            return "EFX", order, ef1_items, new_flag
        elif flag == "EF1":
            return "EF1", order, ef1_items, new_flag
        elif flag == "NotEF1":
            return "NotEF1", order, ef1_items, new_flag
        else:
            return "EF", order, ef1_items, new_flag
    

    def minimax_trade(self):
        """
        Implements a Minimax Trade algorithm to allocate items among agents.
        The goal is to allocate items in a way that minimizes envy and maximizes fairness.

        Returns:
        - numpy.ndarray: The final allocation matrix indicating which agent receives which items.
        """

        # Initialize a list to keep track of unallocated items
        unallocated_items = self.items[:]

        # Select the first item from the unallocated items list
        preferred_item = unallocated_items[0]

        # Allocate the preferred item to the agent that values it the most
        preferred_agent = np.argmax(self.valuation[:, preferred_item])
        self.allocation[preferred_agent][preferred_item] = 1
        unallocated_items.remove(preferred_item)  # Remove the allocated item from the list

        while unallocated_items:
            # Select the next preferred item from the unallocated items list
            preferred_item = unallocated_items[0]

            # Initialize a list to hold potential Pareto exchanges for each agent
            maximal_pareto_exchanges = [None] * self.n

            # Create a copy of the current allocation matrix
            allocation = self.allocation.copy()

            # Iterate through each agent to find potential allocations
            for it_agent in self.agents:
                # Temporarily allocate the preferred item to the current agent
                allocation[it_agent][preferred_item] = 1

                # Find Pareto exchanges for the current allocation and preferred item
                pareto_exchanges = self.find_pareto_exchanges(allocation)
                if pareto_exchanges:
                    # If there are Pareto exchanges, find the maximal Pareto improvements
                    maximal_pareto_exchanges[it_agent] = self.find_maximal_pareto_improvements(pareto_exchanges)

                # Revert the temporary allocation
                allocation[it_agent][preferred_item] = 0

            # Determine the minimal envy bundles and the preferred agent for trading
            minimal_envy_bundles, preferred_agent = self.minimal_envy_bundles(maximal_pareto_exchanges)

            # Perform the trade based on the minimal envy bundles and preferred agent
            self.trade(minimal_envy_bundles, preferred_agent, preferred_item)

            # Remove the allocated item from the list of unallocated items
            unallocated_items.remove(preferred_item)
        
        # Return the final allocation matrix
        final_allocation = self.allocation
        return final_allocation


    
    
    def minimal_envy_bundles(self, maximal_pareto_exchanges):
        """
        Determines the minimal envy bundles from the given minimal Pareto exchanges.

        Parameters:
        - maximal_pareto_exchanges (list): A list of maximal Pareto improvements for each agent.

        Returns:
        - tuple: A tuple containing:
            - minimal_pareto_exchange (tuple): The minimal Pareto exchange (B, C, utils_matrix) that minimizes envy.
            - preferred_item_agent (int): The agent for whom the minimal Pareto exchange was found.
        """

        minimal_pareto_exchange = None  # Initialize the minimal Pareto exchange
        minimal_envy = np.inf  # Initialize the minimal envy to infinity
        num_items = np.sum(np.sum(self.allocation, axis=1)).astype(int)  # Calculate the total number of allocated items

        for i in self.agents:
            if maximal_pareto_exchanges[i]:  # Check if there are Pareto exchanges for agent i
                for (B, C, utils_matrix) in maximal_pareto_exchanges[i]:
                    # Initialize an envy matrix to calculate the envy values
                    envy_matrix = np.zeros((self.n, self.n))
                    for j in self.agents:
                        # Calculate the envy matrix for agent j
                        envy_matrix[j] = utils_matrix[j] - utils_matrix[j][j]

                    # Calculate the maximum envy value
                    envy = np.max(np.max(envy_matrix, axis=1))

                    # Update the minimal envy and the corresponding Pareto exchange if a lower envy is found
                    if envy < minimal_envy:
                        minimal_envy = envy
                        minimal_pareto_exchange = (B, C, utils_matrix)
                        preferred_item_agent = i  # Record the agent for whom the minimal Pareto exchange was found

        return minimal_pareto_exchange, preferred_item_agent


    def trade(self, minimal_pareto_exchange, preferred_agent, preferred_item, partial_allocation=None):
        """
        Executes the trade based on the minimal Pareto exchange to allocate the preferred item.

        Parameters:
        - minimal_pareto_exchange (tuple): The minimal Pareto exchange (B, C, utils_matrix) that minimizes envy.
        - preferred_agent (int): The agent for whom the minimal Pareto exchange was found.
        - preferred_item (int): The item to be allocated.

        Returns:
        - None
        """
        if partial_allocation is not None:
            self.allocation = partial_allocation

        B, C, utils_matrix = minimal_pareto_exchange  # Unpack the minimal Pareto exchange

        if B or C:  # Check if there are items in B or C
            B = list(B)  # Convert B to a list
            C = list(C)  # Convert C to a list
            
            # Update the allocation by removing items in B from agent 0 and items in C from agent 1
            self.allocation[0][B] = 0
            self.allocation[1][C] = 0

            # Allocate items in C to agent 0 and items in B to agent 1
            self.allocation[0][C] = 1
            self.allocation[1][B] = 1

            if preferred_item not in B and preferred_item not in C:
                # If the preferred item is not in B or C, allocate it to the preferred agent
                self.allocation[preferred_agent][preferred_item] = 1
        else:
            # If there are no items in B or C, directly allocate the preferred item to the preferred agent
            self.allocation[preferred_agent][preferred_item] = 1





    def compute_utility_matrix(self, allocation_matrix):
        """
        Computes the utility matrix based on the given allocation matrix.

        Parameters:
        - allocation_matrix (numpy.ndarray): A 2D array where allocation_matrix[i][j] indicates if item j is allocated to agent i.

        Returns:
        - numpy.ndarray: The utility matrix where each element represents the utility an agent derives from the allocation.
        """
        return np.dot(self.valuation, allocation_matrix.T)


    def find_pareto_exchanges(self, allocation_matrix):
        """
        Finds Pareto exchanges for a given allocation matrix and item.

        Parameters:
        - allocation_matrix (numpy.ndarray): A 2D array where allocation_matrix[i][j] indicates if item j is allocated to agent i.
        - item (int): The item being considered for allocation.

        Returns:
        - list: A list of Pareto exchanges, where each exchange is a tuple (B, C, new_utilities_matrix).
                B is the set of items to be exchanged from agent 1 to agent 2.
                C is the set of items to be exchanged from agent 2 to agent 1.
                new_utilities_matrix is the utility matrix after the exchange.
        """

        initial_utilities_matrix = self.compute_utility_matrix(self.allocation)  # Compute initial utility matrix
        initial_utility = np.array(initial_utilities_matrix.diagonal())  # Get initial utilities for each agent
        
        # Get the items currently allocated to agent 1 and agent 2
        items_agent1 = np.where(allocation_matrix[0] == 1)[0]
        items_agent2 = np.where(allocation_matrix[1] == 1)[0]
        
        pareto_exchanges = []

        # Consider all subsets of items to exchange between agents
        for B in chain.from_iterable(combinations(items_agent1, r) for r in range(len(items_agent1) + 1)):
            for C in chain.from_iterable(combinations(items_agent2, r) for r in range(len(items_agent2) + 1)):

                new_allocation = allocation_matrix.copy()  # Create a copy of the current allocation
                
                # Execute the exchange: items in B move from agent 1 to agent 2, and items in C move from agent 2 to agent 1
                for item in B:
                    new_allocation[0, item], new_allocation[1, item] = 0, 1
                for item in C:
                    new_allocation[0, item], new_allocation[1, item] = 1, 0
                
                # Compute the new utility matrix after the exchange
                new_utilities_matrix = self.compute_utility_matrix(new_allocation)
                new_utilities = np.array(new_utilities_matrix.diagonal())  # Get the new utilities for each agent

                # Check if the new allocation is a Pareto improvement
                if all(new_utilities >= initial_utility) and any(new_utilities > initial_utility):
                    pB = set(B)
                    pC = set(C)
                    
                    pareto_exchanges.append((pB, pC, new_utilities_matrix))  # Record the Pareto exchange

        return pareto_exchanges


    def find_maximal_pareto_improvements(self, pareto_exchanges):
        """
        Finds the maximal Pareto improvements from a list of Pareto exchanges.

        Parameters:
        - pareto_exchanges (list): A list of Pareto exchanges, where each exchange is a tuple (B, C, utils_matrix).
                                B is the set of items to be exchanged from agent 1 to agent 2.
                                C is the set of items to be exchanged from agent 2 to agent 1.
                                utils_matrix is the utility matrix after the exchange.

        Returns:
        - list: A list of maximal Pareto exchanges that cannot be further improved.
        """

        maximal_pareto_exchanges = []

        for B, C, utils_matrix in pareto_exchanges:
            is_maximal = True  # Flag to check if the current exchange is maximal
            utility = utils_matrix.diagonal()  # Extract the utility values for the current exchange

            for B1, C1, utils_matrix_1 in pareto_exchanges:
                utility_1 = utils_matrix_1.diagonal()  # Extract the utility values for the comparison exchange

                # Check if the comparison exchange strictly dominates the current exchange
                if np.all(utility_1 >= utility) and any(utility_1 > utility):
                    is_maximal = False  # Current exchange is not maximal
                    break

            if is_maximal:
                # If the current exchange is maximal, add it to the list
                maximal_pareto_exchanges.append((B, C, utils_matrix))

        return maximal_pareto_exchanges

    def allocate_and_trade(self, partial_allocation):
        """
        Implements a Minimax Trade algorithm to allocate items among agents.
        The goal is to allocate items in a way that minimizes envy and maximizes fairness.

        Returns:
        - numpy.ndarray: The final allocation matrix indicating which agent receives which items.
        """

        preferred_item = np.sum(np.sum(partial_allocation, axis=1)).astype(int)  # Calculate the total number of allocated items

        self.allocation = partial_allocation.copy()
        # Initialize a list to hold potential Pareto exchanges for each agent
        maximal_pareto_exchanges = [None] * self.n

        # Create a copy of the current allocation matrix
        allocation = partial_allocation.copy()

        # Iterate through each agent to find potential allocations
        for it_agent in self.agents:
            # Temporarily allocate the preferred item to the current agent
            allocation[it_agent][preferred_item] = 1
        

            # Find Pareto exchanges for the current allocation and preferred item
            pareto_exchanges = self.find_pareto_exchanges(allocation)
            if pareto_exchanges:
                # If there are Pareto exchanges, find the maximal Pareto improvements
                maximal_pareto_exchanges[it_agent] = self.find_maximal_pareto_improvements(pareto_exchanges)

            # Revert the temporary allocation
            allocation[it_agent][preferred_item] = 0

        
        # Determine the minimal envy bundles and the preferred agent for trading
        minimal_envy_bundles, preferred_agent = self.minimal_envy_bundles(maximal_pareto_exchanges)

        # Perform the trade based on the minimal envy bundles and preferred agent
        self.trade(minimal_envy_bundles, preferred_agent, preferred_item)

            # Remove the allocated item from the list of unallocated items
            
        
        # Return the final allocation matrix
        final_allocation = self.allocation
        return final_allocation