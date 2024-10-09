from django.shortcuts import render, redirect
from allocation_app.forms import AlgorithmForm, NumberForm
from django.views.generic import TemplateView


class SelectionsView(TemplateView):
    template_name = 'allocation_app/selections.html'
    
    def post(self, request, *args, **kwargs):
        alg_form = AlgorithmForm(request.POST)
        num_form = NumberForm(request.POST)
        
        if alg_form.is_valid() and num_form.is_valid():
            alg_value = alg_form.cleaned_data['algorithm']
            algorithm_name = dict(AlgorithmForm.base_fields['algorithm'].choices).get(alg_value)
            
            request.session['algorithm'] = alg_value
            request.session['algorithm_name'] = algorithm_name
            request.session['num_agents'] = num_form.cleaned_data['num_agents']
            request.session['num_items'] = num_form.cleaned_data['num_items']
            return redirect('input_preferences')
        
        return render(request, self.template_name, {'alg_form': alg_form, 'num_form': num_form})
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['alg_form'] = AlgorithmForm()
        context['num_form'] = NumberForm()
        return context
    
