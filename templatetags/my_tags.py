from django import template
from django.utils.safestring import mark_safe
from NBAWeb.views import individual_graph

register = template.Library()
@register.simple_tag
def my_tag1(player):
    return mark_safe(individual_graph(player))
@register.simple_tag
def multi(n1, n2):
    if n1 < 0:
        return 0
    return "%.1f" % (n1 * n2)
@register.simple_tag
def limit(data, lower, upper):
    return (data-lower)/(upper-lower)*100