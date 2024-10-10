# allocation_app/views.py

from django.shortcuts import render, redirect
from allocation_app.forms import NumberForm, AlgorithmForm
from django.urls import reverse
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from itertools import permutations
from django.views.generic import TemplateView


class SelectNumbersView(TemplateView):
    template_name = 'allocation_app/select_numbers.html'
    
    def post(self, request, *args, **kwargs):
        form = NumberForm(request.POST)
        if form.is_valid():
            request.session['num_agents'] = form.cleaned_data['num_agents']
            request.session['num_items'] = form.cleaned_data['num_items']
            return redirect('select_algorithm')
        return render(request, self.template_name, {'form': form})
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = NumberForm()
        return context


class SelectAlgorithmView(TemplateView):
    template_name = 'allocation_app/select_algorithm.html'
    
    def post(self, request, *args, **kwargs):
        form = AlgorithmForm(request.POST)
        if form.is_valid():
            request.session['algorithm'] = form.cleaned_data['algorithm']
            return redirect('input_preferences')
        return render(request, self.template_name, {'form': form})
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = AlgorithmForm()
        return context


class InputPreferencesView(TemplateView):
    template_name = 'allocation_app/input_preferences.html'
    
    def post(self, request, *args, **kwargs):
        num_agents = request.session.get('num_agents')
        num_items = request.session.get('num_items')
        agents = [f'Agent {i}' for i in range(1, num_agents + 1)]
        items = [f'Item {i}' for i in range(1, num_items + 1)]
        preferences = [[0] * num_items for _ in range(num_agents)]
        for i, agent in enumerate(agents):
            prefs = request.POST.getlist(agent)
            if len(prefs) != num_items:
                return HttpResponse("Please rank all items for each agent.")
            for j, pref in enumerate(prefs):
                try:
                    preferences[i][j] = float(pref)
                except ValueError:
                    return HttpResponse("Please enter valid integer values for preferences.")
        request.session['preferences'] = preferences
        return redirect('show_allocation')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        num_agents = self.request.session.get('num_agents')
        num_items = self.request.session.get('num_items')
        agents = [f'Agent {i}' for i in range(1, num_agents + 1)]
        items = [f'Item {i}' for i in range(1, num_items + 1)]
        context['agents'] = agents
        context['items'] = items
        return context


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
        agents_indices = list(range(num_agents))
        
        context = {
            'agents': agents,
            'agents_indices': agents_indices,
            'allocation': allocated_items_per_agent,
        }
        return context


