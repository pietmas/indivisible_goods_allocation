# allocation_app/forms.py

from django import forms

ALGORITHM_CHOICES = [
    ('barman', 'Barman Algorithm'),
    ('brute_force', 'Brute Force'),
    ('envy_balance', 'Envy Balance'),
    ('envy_cycle', 'Envy Cycle'),
    ('garg', 'Garg Algorithm'),
    ('generalized_adjusted_winner', 'Generalized Adjusted Winner'),
    ('minimax_envy_trade', 'Minimax Envy Trade'),
    ('mnw', 'Maximum Nash Welfare'),
    ('round_robin', 'Round Robin'),
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
