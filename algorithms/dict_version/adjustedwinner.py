import numpy as np

class AdjustedWinner:
    """
    Class to implement an adjusted-winner procedure.

    This class provides methods to compute envy-free up to one item (EF1) and pareto-optimal allocations for two agents.
    based on the algorithm presented in the paper:
    "Fair allocation of combinations of indivisible goods and chores"
    by Haris Aziz, Ioannis Caragiannis, Ayumi Igarashi, Toby Walsh.
    Link: https://arxiv.org/abs/1807.10684
    """
    def __init__(self, valuations):
        self.valuations = np.array(valuations)  # Ensure valuations are a NumPy array for easy slicing.
        self.n_items = self.valuations.shape[1]
        self.allocation = {0:[],1:[]}  # Initial empty allocation for both agents

    def compute_ratios(self, item_subset):
        """
        Compute and sort the ratios of the valuations of the items in the item_subset.

        This function calculates the ratio of each item's valuation by agent 0 to its valuation by agent 1.
        Items with zero valuation by agent 1 are handled separately to avoid division by zero and are
        sorted by their valuation by agent 0. The final list is sorted with zero-valuation items first,
        followed by items sorted by the computed ratios in descending order.

        Parameters:
        item_subset (list): List of item indices to consider for ratio computation.

        Returns:
        list: Sorted list of item indices based on the computed ratios.
        """

        ratios = []  # List to store the computed ratios of valuations
        # List of items with zero valuation by agent 1
        null_items = [i for i in item_subset if self.valuations[1, i] == 0]
        # List of items with positive valuation by agent 1
        positive_items = [i for i in item_subset if self.valuations[1, i] != 0]

        # Compute the ratios for items with positive valuation by agent 1
        for i in positive_items:
            ratios.append(self.valuations[0, i] / self.valuations[1, i])

        # Sort the items based on the computed ratios in descending order
        sorted_items = sorted(positive_items, key=lambda i: ratios[positive_items.index(i)], reverse=True)
        # Sort the items with zero valuation by agent 1 based on their valuation by agent 0
        sorted_null_items = sorted(null_items, key=lambda i: self.valuations[0, i], reverse=True)

        # Combine the sorted null items and sorted positive items
        return sorted_null_items + sorted_items

    def is_ef1(self, first_agent, second_agent):
        """
        Check if the allocation between the two agents is envy-free up to one item (EF1).

        This function checks if the allocation for `first_agent` is EF1 with respect to `second_agent`.
        EF1 means that `first_agent` should not envy `second_agent` after the possible removal of any single item from
        `second_agent`'s allocation.

        Parameters:
        first_agent (int): Index of the first agent.
        second_agent (int): Index of the second agent.

        Returns:
        bool: True if the allocation is EF1 for `first_agent` with respect to `second_agent`, False otherwise.
        """
        # Calculate the total valuation of first_agent's own items
        if len(self.allocation[first_agent]) == 0:
            valuation_looser_for_own_items = 0
        else:
            valuation_looser_for_own_items = sum(self.valuations[first_agent, i] for i in self.allocation[first_agent])
        
        # Calculate the total valuation of second_agent's items by first_agent's valuation
        if len(self.allocation[second_agent]) == 0:
            valuation_looser_for_other_items = 0
        else:
            valuation_looser_for_other_items = sum(self.valuations[first_agent, i] for i in self.allocation[second_agent])
        
        # Check if first_agent does not envy second_agent's allocation
        if valuation_looser_for_own_items >= valuation_looser_for_other_items:
            return True
        else:
            # Check if removing any single item from second_agent's allocation removes the envy
            for i in self.allocation[second_agent]:
                if valuation_looser_for_own_items >= valuation_looser_for_other_items - self.valuations[first_agent, i]:
                    return True
        
        return False
    
    def allocate_items(self, item_subset=None):
        """
        Allocate items to agents while ensuring the allocation is envy-free up to one item (EF1).

        This function allocates items to agents, ensuring that the resulting allocation is EF1.
        If an item_subset is provided, only those items are considered for allocation.
        The function returns all possible EF1 allocations.

        Parameters:
        item_subset (list, optional): List of item indices to consider for allocation. 
                                      If None, all items are considered.

        Returns:
        list: List of all possible EF1 allocations.
        """
        if item_subset is None:
            # If no subset is provided, consider all items
            item_subset = list(range(self.n_items))
        
        # Compute the ratios and sort items accordingly
        items = self.compute_ratios(item_subset)
        
        all_allocation = []  # List to store all possible EF1 allocations
        allocated_utility = [0, 0]  # Initialize total utilities for both agents

        # Allocate all items to agent 1 initially
        self.allocation[1] = items[:]
        self.allocation[0] = []

        for i in items:
            # Check if the allocation is EF1 for both agents
            if self.is_ef1(0, 1):
                return self.allocation
            else:
                # If not EF1, remove item from agent 1 and allocate to agent 0
                self.allocation[1].remove(i)
                self.allocation[0].append(i)


        return self.allocation
        


    def get_allocation(self, item_subset=None):
        """
        Get the final allocation of items ensuring EF1 (envy-free up to one item).

        This function resets the current allocation, then allocates the items to the agents using
        the `allocate_items` method. If an item_subset is provided, only those items are considered
        for allocation.

        Parameters:
        item_subset (list, optional): List of item indices to consider for allocation.
                                      If None, all items are considered.

        Returns:
        list: List of all possible EF1 allocations.
        """
        # Resets the allocation for both agents
        self.allocation = {0: [], 1: []}
        
        # Call the allocate_items method to get the final allocation for the given subset of items
        all_allocation = self.allocate_items(item_subset=item_subset)
        print(all_allocation)
        # Return the list of all possible EF1 allocations
        return all_allocation