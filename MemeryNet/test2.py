import torch
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(1))
        self.register_parameter('param1', self.param)

    def forward(self, x):
        self.x = x
        x = self.param * x
        return x


model = MyModel()
print(dict(model.named_parameters()))

out = model(torch.randn(1))
out.backward()

print(model.param.grad)
print(model.param1.grad)

# next iteration
model.zero_grad()
model.register_parameter('param2', nn.Parameter(torch.randn(1)))
print(dict(model.named_parameters()))

out = model(torch.randn(1))
out.backward()

print(model.param.grad)
print(model.param1.grad)