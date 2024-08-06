import numpy as np

class EnvyCycleElimination:
    """
    Class to implement the envy-cycle elimination algorithm presented in https://dl.acm.org/doi/10.1145/988772.988792

    This class provides methods to compute envy-free up to one item (EF1) allocation 
    based on the algorithm presented in the paper:
    "On approximately fair allocations of indivisible goods"
    by  R. J. Lipton, E. Markakis, E. Mossel, A. Saberi.
    Link: https://dl.acm.org/doi/10.1145/988772.988792.
    """
    def __init__(self, n, m, v):
        """
        Initialize the EnvyCycleElimination class.

        Parameters:
        n (int): Number of agents.
        m (int): Number of goods.
        v (np.array): Valuation matrix where v[i][j] is the valuation of agent i for good j.
        """
        
        self.n_agents = n  # number of agents
        self.m_items = m  # number of goods
        self.valuations = v  # valuations of agents for goods
        self.allocation = np.zeros((self.n_agents, self.m_items), dtype=int)  # initial empty allocation
        self.unallocated_items = set(range(m))  # set of all goods
        self.envy_graph = {i: [] for i in range(self.n_agents)}  # envy graph

    def compute_envy_graph(self):
        """
        Construct the envy graph based on current allocation.

        Returns:
        dict: A dictionary representing the envy graph where keys are agents and values are lists of agents they envy.
        """
        # Initialize empty envy graph
        utility_matrix = np.dot(self.valuations, self.allocation.T)  # utility matrix of agents for goods
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j:
                    # If agent i values agent j's allocation more than their own
                    if utility_matrix[i][j] > utility_matrix[i][i]:
                        # Agent i envies agent j 
                        self.envy_graph[j].append(i)  
    
    def reset_envy_graph(self):
        """
        Reset the envy graph.
        """
        self.envy_graph = {i: [] for i in range(self.n_agents)}

    def detect_cycle(self):
        """
        Detect a cycle in the envy graph.

        Parameters:
        envy_graph (dict): The envy graph.

        Returns:
        list: A list representing the cycle detected, empty if no cycle found.
        """
        visited = set()  # Set to keep track of visited nodes
        stack = []  # Stack to keep track of the current path

        def visit(node):
            if node in visited:
                if node in stack:
                    # Return the cycle found
                    return stack[stack.index(node):]  
                return []
            # Mark node as visited and add to the current path
            visited.add(node)  
            stack.append(node) 


            for neighbor in self.envy_graph[node]:
                # Recursively visit neighbors
                cycle = visit(neighbor)  
                if cycle:
                    # Return the cycle if found
                    return cycle 
            # Remove node from the current path 
            stack.pop()  
            return []

        for node in self.envy_graph:
            cycle = visit(node)
            if cycle:
                # Return the cycle if found
                return cycle
        # Return an empty list if no cycle found 
        return []  

    def find_envy_cycle(self):
        """
        Find an envy cycle in the current allocation.

        Returns:
        list: A list representing the cycle detected.
        """
        # Construct the envy graph
        self.compute_envy_graph() 
        # Detect a cycle in the envy graph
        cycle = self.detect_cycle() 
         
        return cycle

    def envy_cycle_elimination(self):
        """
        Perform envy cycle elimination to find an EF1 allocation.

        Returns:
        np.array: The allocation matrix where rows represent agents and columns represent goods.
        """
        for ell in range(1, self.m_items + 1):
            while True:
                self.reset_envy_graph()
                # Generate the envy graph
                self.compute_envy_graph()
                # Find agents who are not envied by any other agent
                unenvied_agents = [i for i in range(self.n_agents) if not self.envy_graph[i]]  
                if len(unenvied_agents) == self.n_agents:
                    break  # Stop if all agents are unenvied

                # Find an envy cycle
                cycle = self.find_envy_cycle()


                if not cycle:
                    break  # Stop if no cycle is found

                # Rotate allocations according to the cycle
                d = len(cycle)
                AC = {}
                for j in range(d):
                    next_agent = cycle[(j + 1) % d]
                    # Store the allocation of the next agent in the cycle
                    AC[cycle[j]] = self.allocation[next_agent]

                for i in cycle:
                    # Update the allocation for the current agent in the cycle
                    self.allocation[i] = AC[i]
                    

            # Select an unenvied agent and allocate the most valuable good
            unenvied_agents = [i for i in range(self.n_agents) if not self.envy_graph[i]]  # Re-evaluate unenvied agents after cycle elimination
            if not unenvied_agents:
                break  # Stop if no unenvied agents are left
            i = unenvied_agents[0]

            # Find the most valuable good for the unenvied agent, allocate it, and remove it from the set of goods
            g_star = max(self.unallocated_items, key=lambda g: self.valuations[i][g]) 
            self.allocation[i][g_star] = 1
            self.unallocated_items.remove(g_star)  

            allocation = self.allocation

        return allocation  # Return the allocation matrix
