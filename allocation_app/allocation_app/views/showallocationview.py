from django.http import HttpResponse
from django.shortcuts import redirect
from django.views.generic import TemplateView
import numpy as np


class ShowAllocationView(TemplateView):
    template_name = 'allocation_app/show_allocation.html'
    
    def get_context_data(self, **kwargs):
        algorithm = self.request.session.get('algorithm')
        num_agents = self.request.session.get('num_agents')
        num_items = self.request.session.get('num_items')
        preferences = self.request.session.get('preferences')


        if not algorithm or not preferences:
            return {}
        
        # Import your algorithms dynamically
        import sys
        sys.path.append('..')  # Adjust the path as needed

        # Prepare data in the format required by your algorithms
        # Example assumes preferences are dictionaries
        if algorithm == 'barman':
            from algorithms.barman import Barman
            allocations, prices = Barman(num_agents, num_items, np.array(preferences)).run_algorithm()
        elif algorithm == 'brute_force':
            from algorithms.brute_force import BruteForce
            allocations = BruteForce(num_agents, num_items, np.array(preferences)).compute_ef1_and_po_allocations()
        elif algorithm == 'envy_balance':
            from algorithms.envy_balancing import EnvyBalancing
            allocations = EnvyBalancing(np.array(preferences)).run_algorithm()
        elif algorithm == 'envy_cycle':
            from algorithms.envy_cycle import EnvyCycleElimination
            allocations = EnvyCycleElimination(num_agents, num_items, np.array(preferences)).envy_cycle_elimination()
        elif algorithm == 'garg':
            from algorithms.garg import GargAlgorithm
            allocations, price = GargAlgorithm(num_agents, num_items, np.array(preferences)).run_algorithm()
        elif algorithm == 'generalized_adjusted_winner':
            from algorithms.generalized_adjusted_winner import GeneralizedAdjustedWinner
            allocations = GeneralizedAdjustedWinner(np.array(preferences)).get_allocation()
        elif algorithm == 'minimax_envy_trade':
            from algorithms.minmaxenvy_trade import MinimaxTrade
            allocations = MinimaxTrade(num_agents, num_items, np.array(preferences)).minimax_trade()
        elif algorithm == 'mnw':
            from algorithms.mnw import maximize_nash_welfare_bruteforce
            allocations = maximize_nash_welfare_bruteforce(num_agents, num_items, np.array(preferences))
        elif algorithm == 'round_robin':
            from algorithms.round_robin import round_robin_allocation
            allocations = round_robin_allocation(num_agents, num_items, np.array(preferences))
        else:
            return HttpResponse("Selected algorithm is not implemented.")
        
        # Restore the path
        sys.path.remove('..')
        
        if not isinstance(allocations, list):
            allocations = [allocations]

        # Prepare context data for rendering the allocation
        agents = [f'Agent {i}' for i in range(1, num_agents + 1)]
        items = [f'Item {i}' for i in range(1, num_items + 1)]
        all_allocations = []
        for allocation in allocations:
            allocated_items_per_agent = []
            for agent_index in range(num_agents):
                allocated_items = []
                for item_index in range(num_items):
                    if allocation[agent_index][item_index] == 1:
                        allocated_items.append(items[item_index])
                allocated_items_per_agent.append(allocated_items)
            all_allocations.append(allocated_items_per_agent)
        
        algorithm = self.request.session.get('algorithm_name')


        context = {
            'algorithm': algorithm,
            'items': items,
            'agents': agents,
            'items_indices': list(range(num_items)),
            'agents_indices': list(range(num_agents)),
            'preferences': preferences,
            'allocations': all_allocations,
        }
        return context