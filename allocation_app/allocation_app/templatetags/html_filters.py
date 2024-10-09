from django import template

register = template.Library()

@register.filter
def get_item(array, key):
    return array[key]

@register.filter
def index(sequence, position):
    try:
        return sequence[position]
    except IndexError:
        return None
