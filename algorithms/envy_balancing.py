import numpy as np
from utils.check import Checker

class EnvyBalancing:
    """
    A class to implement an envy-free allocation algorithm.

    This class attempts to allocate items among agents such that the final allocation is envy-free,
    meaning that no agent prefers another agent's allocation over their own.

    The algorithm is proposed in:
    "Achieving a fairer future by changing the past"
    by Jiafan He, Ariel D. Procaccia, Alexandros Psomas, David Zeng
    Link: https://dl.acm.org/doi/10.5555/3367032.3367082
    
    Attributes:
        n_agents (int): Number of agents.
        m_items (int): Number of items.
        items (list of int): List of item indices.
        valuations (numpy.ndarray): Valuation matrix where each element represents the value assigned by an agent to an item.
        Ct (list of tuple): A list of tuples representing candidate allocations at each round.
        s (int): The last round that was envy-free.
        allocation (numpy.ndarray): A matrix representing the current allocation of items to agents.
    """
    def __init__(self, valuations):
        """
        Initializes the EnvyBalancing class with the given valuations matrix.

        Parameters:
            valuations (numpy.ndarray): Valuation matrix where each element represents the value assigned by an agent to an item.
        """
        self.n_agents = len(valuations)  # Number of agents
        self.m_items = len(valuations[0])  # Number of items
        self.items = list(range(self.m_items))  # List of item indices
        self.valuations = valuations  # Valuation matrix
        self.Ct = [(set(), set())]  # Candidate allocations at each round
        self.s = 0  # Last round that was envy-free
        self.allocation = np.zeros((self.n_agents, self.m_items), dtype=int)  # Allocation matrix initialized to zeros

    def run_algorithm(self):
        """
        Runs the envy-free allocation algorithm.

        This method iteratively allocates items to agents, ensuring that the final allocation is envy-free.
        It checks for envy at each step and adjusts the allocations accordingly.

        Returns:
            numpy.ndarray: The final envy-free allocation matrix.
        """
        # Create a copy of the item list to keep track of unallocated items.
        unallocated_items = self.items.copy()

        # Iterate through each item to allocate.
        for t in range(self.m_items):
            gt = unallocated_items.pop(0)  # Get the next unallocated item
            a1, a2 = self.compute_Cst(t)  # Compute current candidate sets for allocation
            ct_1, ct_2 = self.Ct[t]  # Get the candidate allocation at round t

            # Check if agent 1 does not envy agent 2 in the current allocation
            if self.is_unenvied(a1, a2):
                Ct_new = (ct_1.union({gt}), ct_2)  # Add item to agent 1's candidate set if no envy
            else:
                Ct_new = (ct_1, ct_2.union({gt}))  # Otherwise, add item to agent 2's candidate set

            # Update the candidate allocations for the next round
            self.Ct.append(Ct_new)
            self.compute_allocation()  # Update the allocation matrix based on current candidates

            # Compute new candidate sets for the next round
            a_1, a_2 = self.compute_Cst(t + 1)

            # Check if the current allocation causes any envy
            if self.do_agents_envy(a_1, a_2):
                # If envy is detected, swap the allocations between the two agents
                self.allocation[0][list(a_1)] = 0
                self.allocation[1][list(a_2)] = 0

                self.allocation[0][list(a_2)] = 1
                self.allocation[1][list(a_1)] = 1
                self.Ct = self.Ct[:-1]  # Remove the last candidate allocation
                Ct_new = (set(np.where(self.allocation[0] == 1)[0]), set(np.where(self.allocation[1] == 1)[0]))
                self.Ct.append(Ct_new)  # Update the candidate allocation list

            # Check if the current allocation is envy-free
            if self.is_envy_free(a_1, a_2):
                self.s = t + 1  # Update the last envy-free round

            # Calculate the number of items allocated so far
            num_items = np.sum(np.sum(self.allocation, axis=0))

            # Check if the current allocation is Pareto optimal using the Checker utility
            check = Checker(self.n_agents, num_items, self.valuations[:, :num_items], self.allocation[:, :num_items])
            if not check.is_pareto_optimal():
                print(f"Valuations: \n{self.valuations}\n")
                print(f"Allocation: \n{self.allocation}\n")
                raise ValueError("The allocation is not Pareto optimal")

        # Return the final allocation matrix
        return self.allocation

    def is_unenvied(self, a1, a2):
        """
        Checks if agent 1 does not envy agent 2.

        Parameters:
            a1 (set): Set of items allocated to agent 1.
            a2 (set): Set of items allocated to agent 2.

        Returns:
            bool: True if agent 1 does not envy agent 2, False otherwise.
        """
        return sum([self.valuations[1][j] for j in a2]) >= sum([self.valuations[1][i] for i in a1])

    def do_agents_envy(self, a1, a2):
        """
        Checks if any agent envies the other in the current allocation.

        Parameters:
            a1 (set): Set of items allocated to agent 1.
            a2 (set): Set of items allocated to agent 2.

        Returns:
            bool: True if either agent envies the other, False otherwise.
        """
        return (sum([self.valuations[0][j] for j in a2]) > sum([self.valuations[0][i] for i in a1])) and \
               (sum([self.valuations[1][j] for j in a1]) > sum([self.valuations[1][i] for i in a2]))

    def is_envy_free(self, a1, a2):
        """
        Checks if the current allocation is envy-free.

        Parameters:
            a1 (set): Set of items allocated to agent 1.
            a2 (set): Set of items allocated to agent 2.

        Returns:
            bool: True if the allocation is envy-free, False otherwise.
        """
        return sum([self.valuations[0][j] for j in a2]) <= sum([self.valuations[0][i] for i in a1]) and \
               sum([self.valuations[1][j] for j in a1]) <= sum([self.valuations[1][i] for i in a2])
    
    def compute_Cst(self, t):
        """
        Computes the candidate sets for allocation at round t.

        Parameters:
            t (int): The current round.

        Returns:
            tuple: Two sets representing the candidate allocations for the two agents.
        """
        if t == 0:
            return (set(), set())  # Return empty sets for the first round
        else:
            Cst_1, Cst_2 = self.Ct[t]  # Get the candidate sets from the last round
            # Filter out items that were allocated in the last envy-free round
            Cst_1 = [i for i in Cst_1 if self.s < i + 1]
            Cst_2 = [i for i in Cst_2 if self.s < i + 1]
            return Cst_1, Cst_2

    def compute_allocation(self):
        """
        Updates the allocation matrix based on the current candidate sets.

        This method updates the allocation matrix by assigning items to agents according to the current candidate sets.

        Returns:
            None
        """
        ct_1, ct_2 = self.Ct[-1]  # Get the last candidate sets
        if ct_1:
            self.allocation[0][list(ct_1)] = 1  # Assign items in ct_1 to agent 1
        if ct_2:
            self.allocation[1][list(ct_2)] = 1  # Assign items in ct_2 to agent 2
