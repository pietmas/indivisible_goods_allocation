from django.http import HttpResponse
from django.shortcuts import redirect
from django.views.generic import TemplateView


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
        from algorithms import brute_force, round_robin  # Import other algorithms as needed
        
        # Prepare data in the format required by your algorithms
        # Example assumes preferences are dictionaries
        
        if algorithm == 'brute_force':
            allocation = brute_force.allocate(preferences)
        elif algorithm == 'round_robin':
            allocation = round_robin.round_robin_allocation(num_agents, num_items, preferences)
        else:
            return HttpResponse("Selected algorithm is not implemented.")
        
        # Prepare context data for rendering the allocation
        agents = [f'Agent {i}' for i in range(1, num_agents + 1)]
        items = [f'Item {i}' for i in range(1, num_items + 1)]
        allocated_items_per_agent = []
        for agent_index in range(num_agents):
            allocated_items = []
            for item_index in range(num_items):
                if allocation[agent_index][item_index] == 1:
                    allocated_items.append(items[item_index])
            allocated_items_per_agent.append(allocated_items)
        
        algorithm = self.request.session.get('algorithm_name')


        context = {
            'algorithm': algorithm,
            'items': items,
            'agents': agents,
            'items_indices': list(range(num_items)),
            'agents_indices': list(range(num_agents)),
            'preferences': preferences,
            'allocation': allocated_items_per_agent,
        }
        return context