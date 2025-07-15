import torch
from IPython.display import display, Math
a = torch.tensor([2.0],requires_grad=True) #automatic differentiation
b = torch.tensor([3.0],requires_grad=True) #automatic differentiation
c = torch.tensor([5.0],requires_grad=True) #automatic differentiation
d = torch.tensor([10.0],requires_grad=True) #automatic differentiation
u = a*b
t = torch.log10(d)
v = t*c
t.retain_grad()
e = u+v
print(e)
e.backward()
display(Math(fr'\frac{{\partial e}}{{\partial a}} = {a.grad.item()}'))
print()
display(Math(fr'\frac{{\partial e}}{{\partial b}} = {b.grad.item()}'))
print()
display(Math(fr'\frac{{\partial e}}{{\partial c}} = {c.grad.item()}'))
print()
display(Math(fr'\frac{{\partial e}}{{\partial d}} = {d.grad.item()}'))