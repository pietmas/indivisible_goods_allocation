# allocation_app/forms.py

from django import forms

ALGORITHM_CHOICES = [
    ('round_robin', 'Round Robin'),
    ('envy_cycle', 'Envy Cycle'),
    # Add other algorithms as needed
]

class NumberForm(forms.Form):
    num_agents = forms.IntegerField(
        min_value=1,
        label='Number of Agents',
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    num_items = forms.IntegerField(
        min_value=1,
        label='Number of Items',
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

class AlgorithmForm(forms.Form):
    algorithm = forms.ChoiceField(
        choices=ALGORITHM_CHOICES,
        label='Allocation Algorithm',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
