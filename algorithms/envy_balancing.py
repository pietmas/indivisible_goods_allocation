import numpy as np
from utils.check import Checker
class EnvyBalancing:
    def __init__(self, valuations):
        self.n_agents = len(valuations)
        self.m_items = len(valuations[0])
        self.items = list(range(self.m_items))
        self.valuations = valuations
        self.Ct = [(set(), set())]
        self.s = 0  # last round that was envy-free
        self.allocation = np.zeros((self.n_agents, self.m_items), dtype=int)

    def run_algorithm(self):
        # Iterating through items
        unallocated_items = self.items.copy()

        for t in range(self.m_items):
            gt = unallocated_items.pop(0)
            a1, a2 = self.compute_Cst(t)
            ct_1, ct_2 = self.Ct[t]
            # Check if a1 is unenvied in the last allocation
            if self.is_unenvied(a1, a2):
                Ct_new = (ct_1.union({gt}), ct_2)
            else:
                Ct_new = (ct_1, ct_2.union({gt}))

            # Update the candidate allocations
            self.Ct.append(Ct_new)
            self.compute_allocation()
            # Update 
            a_1, a_2 = self.compute_Cst(t + 1)
            # Check if the current allocation is envy-free
            if self.do_agents_envy(a_1, a_2):
                self.allocation[0][list(a_1)] = 0
                self.allocation[1][list(a_2)] = 0

                self.allocation[0][list(a_2)] = 1
                self.allocation[1][list(a_1)] = 1
                self.Ct = self.Ct[:-1]
                Ct_new = (set(np.where(self.allocation[0] == 1)[0]), set(np.where(self.allocation[1] == 1)[0]))
                self.Ct.append(Ct_new)
            
            if self.is_envy_free(a_1, a_2):
                self.s = t + 1

            num_items = np.sum(np.sum(self.allocation, axis=0))

            check = Checker(self.n_agents, num_items, self.valuations[:, :num_items], self.allocation[:, :num_items])
            if not check.is_pareto_optimal():
                print(f"Valuations: \n{self.valuations}\n")
                print(f"Allocation: \n{self.allocation}\n")
                raise ValueError("The allocation is not pareto optimal")
                

            # Update the last envy-free round


        return self.allocation

    def is_unenvied(self, a1, a2):
        return sum([self.valuations[1][j] for j in a2]) >= sum([self.valuations[1][i] for i in a1])

    def do_agents_envy(self, a1, a2):
        return (sum([self.valuations[0][j] for j in a2]) > sum([self.valuations[0][i] for i in a1])) and \
        (sum([self.valuations[1][j] for j in a1]) > sum([self.valuations[1][i] for i in a2]))

    def is_envy_free(self, a1, a2):
        return sum([self.valuations[0][j] for j in a2]) <= sum([self.valuations[0][i] for i in a1]) and \
        sum([self.valuations[1][j] for j in a1]) <= sum([self.valuations[1][i] for i in a2])
    
    def compute_Cst(self, t):
        if t == 0:
            return (set(), set())
        else:
            Cst_1, Cst_2 = self.Ct[t]
            Cst_1 = [i for i in Cst_1 if self.s < i + 1]
            Cst_2 = [i for i in Cst_2 if self.s < i + 1]
            return Cst_1, Cst_2

    def compute_allocation(self):
        ct_1, ct_2 = self.Ct[-1]
        if ct_1:
            self.allocation[0][list(ct_1)] = 1
        if ct_2:
            self.allocation[1][list(ct_2)] = 1 
        

            