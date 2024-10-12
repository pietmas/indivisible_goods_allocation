from django.views.generic import TemplateView

class HomeView(TemplateView):
    template_name = 'allocation_app/home.html'
    