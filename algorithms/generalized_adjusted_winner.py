import numpy as np

class GeneralizedAdjustedWinner:
    """
    Class to implement an adjusted-winner procedure, it generates all the EF1 and PO allocations 
    for two agents based, computing the algorithm for both agents.

    This class provides methods to compute envy-free up to one item (EF1) and pareto-optimal allocations for two agents.
    based on the algorithm presented in the paper:
    "Fair allocation of combinations of indivisible goods and chores"
    by Haris Aziz, Ioannis Caragiannis, Ayumi Igarashi, Toby Walsh.
    Link: https://arxiv.org/abs/1807.10684
    """
    def __init__(self, valuations, items=None):
        self.valuations = valuations
        self.agents = [0, 1]
        
        self.items = self.initialize_items(items)    

        self.num_agents = 2
        self.num_items = len(self.valuations[0])
        self.allocation = np.zeros((self.num_agents, self.num_items), dtype=int)
        self.winner = None
        self.looser = None

    def initialize_items(self, items):
        if items is None:
            return list(range(len(self.valuations[0])))
        else:
            return items

    def ordered_ratios(self, allocated_items_to_winner):
        """
        Calculates and orders the items allocated to the winner based on the ratio of the loser's valuation 
        to the winner's valuation for each item. The items are sorted in descending order of these ratios.

        Parameters:
        allocated_items_to_winner (list): A list of item indices that have been allocated to the winner.

        Returns:
        list: A list of item indices sorted in descending order based on the ratio of the loser's 
            valuation to the winner's valuation.
        """
        # Initialize an empty list to store the ratios of the loser's valuation to the winner's valuation for each item
        ratios = []
        
        # Calculate the ratio for each item allocated to the winner
        for i in allocated_items_to_winner:
            # Append the ratio of the loser's valuation to the winner's valuation for the current item
            # Note: Assuming valuations for both winner and looser are not zero to avoid division by zero error
            ratios.append(self.valuations[self.looser, i] / self.valuations[self.winner, i])
        
        # Sort the allocated items in descending order based on the calculated ratios
        ordered_allocation = sorted(allocated_items_to_winner, key=lambda i: ratios[np.where(allocated_items_to_winner == i)[0][0]], reverse=True)
        
        # Return the ordered list of allocated items
        return ordered_allocation

    
    def is_EF1(self, first_agent, second_agent):
        '''
        Check if the current allocation is Envy-Free up to one item (EF1) between two specified agents.

        Parameters:
        first_agent (int): The index of the first agent.
        second_agent (int): The index of the second agent.

        Returns:
        bool: True if the allocation is EF1 for the first agent with respect to the second agent, False otherwise.
        '''
  
        # Calculate the utility matrix by multiplying the valuations with the transpose of the allocation matrix
        utility_matrix = np.dot(self.valuations, self.allocation.T)
        # Check if the first agent's utility from their own allocation is at least as much as their utility from the second agent's allocation
        if utility_matrix[first_agent][first_agent] >= utility_matrix[first_agent][second_agent]:
            return True
        else:
            # If the first agent envies the second agent, check if this envy can be reduced by removing any single item from the second agent's allocation
            for item in np.where(self.allocation[second_agent] == 1)[0]:
                if utility_matrix[first_agent][first_agent] >= utility_matrix[first_agent][second_agent] - self.valuations[first_agent][item]:
                    # If the envy can be reduced by removing a single item, return False (allocation is not EF1)
                    return True
        
        # If none of the conditions above are met, return True (allocation is EF1)
        return False


    def allocation_reset(self):
        """
        Resets the allocation matrix to its initial state where no items are allocated to any agents.
        This function is useful for reinitializing the allocation process.

        Parameters:
        None

        Returns:
        None
        """
        # Reset the allocation matrix to zeros, with dimensions corresponding to the number of agents and items
        self.allocation = np.zeros((self.num_agents, self.num_items), dtype=int)


    def allocate(self, items=None):
        """
        Allocate items to agents using the Adjusted Winner procedure.
        
        This function allocates items to the winner initially and then adjusts the allocation
        by transferring items to the loser in a way that maintains the EF1 (Envy-Free up to one item) property.

        Parameters:
        None

        Returns:
        list: A list of allocation matrices representing all intermediate allocations that were EF1.
        """
        # Initialize a list to store all EF1 allocations
        all_allocation = []
        if items is None:
            items = self.items
        # Initially allocate items to the winner based on their positive valuations
        for item in items:
            if self.valuations[self.winner, item] > 0:
                self.allocation[self.winner, item] = 1
            else:
                self.allocation[self.looser, item] = 1
        
        # Sort the allocated items based on the ratio of the loser's valuation to the winner's valuation
        ordered_allocation = self.ordered_ratios(np.where(self.allocation[self.winner] == 1)[0])
        
        # Adjust the allocation to ensure EF1 property
        for i in ordered_allocation:
            # Check if the current allocation is EF1 for both agents
            if self.is_EF1(self.looser, self.winner) and self.is_EF1(self.winner, self.looser):

                # If it is EF1, copy the current allocation and store it
                alloc = self.allocation.copy()
                all_allocation.append(alloc)
                # Transfer the item from the winner to the loser
                break
            else:
                # If not EF1, just transfer the item from the winner to the loser
                self.allocation[self.winner][i] = 0
                self.allocation[self.looser][i] = 1

        # Return all intermediate EF1 allocations
        return all_allocation

    
    def get_allocation(self, w_and_l=None):
        """
        Generates and returns all possible EF1 allocations by considering different winner and loser pairs.

        Parameters:
        w_and_l (list of tuples): A list of (winner, loser) pairs. If not provided, defaults to [(0, 1), (1, 0)].

        Returns:
        list: A list of allocation matrices representing all possible EF1 allocations.
        """
        # Set default winner and loser pairs if not provided
        if w_and_l is None:
            w_and_l = [(0, 1), (1, 0)]
        
        # Initialize a list to store all EF1 allocations
        all_allocations = []
        
        # Iterate through each (winner, loser) pair
        for w, l in w_and_l:
            # Set the current winner and loser
            self.winner = w
            self.looser = l
            
            # Allocate items using the current winner and loser
            allocations = self.allocate()
            
            # Append all intermediate EF1 allocations to the all_allocations list
            for allocation in allocations:
                all_allocations.append(allocation)
            
            # Reset the allocation matrix for the next (winner, loser) pair
            self.allocation_reset()

        # Return all EF1 allocations
        return all_allocations

                              
        