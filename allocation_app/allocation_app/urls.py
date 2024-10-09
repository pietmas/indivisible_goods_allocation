# allocation_app/urls.py

from django.urls import path
from .views.selectionsview import SelectionsView
from .views.inputpreferencesview import InputPreferencesView
from .views.showallocationview import ShowAllocationView

urlpatterns = [
    path('', SelectionsView.as_view(), name='selections'),
    path('input_preferences/', InputPreferencesView.as_view(), name='input_preferences'),
    path('show_allocation/', ShowAllocationView.as_view(), name='show_allocation'),
]
