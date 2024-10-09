from allocation_app.forms import NumberForm
from django.shortcuts import render, redirect
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