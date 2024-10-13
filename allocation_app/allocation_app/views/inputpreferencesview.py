from django.http import HttpResponse
from django.shortcuts import redirect
from django.views.generic import TemplateView

class InputPreferencesView(TemplateView):
    template_name = 'allocation_app/input_preferences.html'
    
    def post(self, request, *args, **kwargs):
        # Get the number of agents and items from the session
        num_agents = request.session.get('num_agents')
        num_items = request.session.get('num_items')
        agents = [f'Agent {i}' for i in range(1, num_agents + 1)]
        items = [f'Item {i}' for i in range(1, num_items + 1)]

        # Get the preferences from the form
        preferences = [[0] * num_items for _ in range(num_agents)]
        for i, agent in enumerate(agents):
            
            for j, item in enumerate(items):
                value = request.POST.getlist(f"preferences-{item}-{agent}")
                try:
                    preferences[i][j] = float(*value)
                except ValueError:
                    return HttpResponse("Please enter valid integer values for preferences.")
        return redirect('show_allocation')
    
    def get_context_data(self, **kwargs):
        # Get the number of agents and items from the session
        context = super().get_context_data(**kwargs)
        num_agents = self.request.session.get('num_agents')
        num_items = self.request.session.get('num_items')


        agents = [f'Agent {i}' for i in range(1, num_agents + 1)]
        items = [f'Item {i}' for i in range(1, num_items + 1)]
        algorithm = self.request.session.get('algorithm_name')
        
        # Add the agents, items, and algorithm to the context
        context['algorithm'] = algorithm
        context['agents'] = agents
        context['items'] = items
        return context