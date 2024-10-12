# allocation_app/views/algorithmview.py

from django.views.generic import TemplateView
from django.http import Http404
import os
import json
from django.conf import settings

class AlgorithmDetailView(TemplateView):
    template_name = 'allocation_app/algorithm_detail.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        algorithm_name = self.kwargs.get('algorithm_name')  # Get the algorithm name from the URL

        # Load the JSON file from the static folder
        json_file_path = os.path.join(settings.BASE_DIR, 'allocation_app', 'static', 'json', 'algorithm_info.json')

        try:
            with open(json_file_path, 'r') as json_file:
                algorithms_info = json.load(json_file)
        except FileNotFoundError:
            raise Http404("Algorithm data not found")
        # Search for the algorithm in the JSON data
        algorithm_data = None
        for key, algorithm in algorithms_info.items():
            if algorithm['name'] == algorithm_name:
                algorithm_data = algorithm
                break

        if not algorithm_data:
            raise Http404(f"Algorithm '{algorithm_name}' not found")

        # Add the algorithm data to the context
        context['algorithm_info'] = algorithm_data
        return context

        
