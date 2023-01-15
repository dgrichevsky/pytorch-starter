# 1) Design model (input, output size, forward pass)
# 2) Construct the loss and optimizer
# 3) Training loop
    # - forward pass: compute prediction
    # - backward pass: gradient descent
    # - update weights
    
import torch
import torch.nn as nn

A = torch.tensor([[1], [2],[3], [4]], dtype=torch.float32)
B = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
W = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self,x):
        return self.lin(x)


learning_rate = 0.01
n_iters = 100
n_samples, n_features = A.shape
X_test = torch.tensor([5], dtype=torch.float32)
input_size, output_size = n_features, n_features
print(n_samples, n_features)
loss = nn.MSELoss()
# model = nn.Linear(input_size, output_size)

model = LinearRegression(input_size, output_size)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(f'Prediction before training f(5) = {model(X_test).item():.3f}')

for epoch in range(n_iters):
    # prediction = forward pass
    B_pred = model(A)
    #loss
    L = loss(B, B_pred)
    # gradients = backward pass
    L.backward() # dL/dW
    optimizer.step()
    optimizer.zero_grad() 
    
    if epoch % 1 == 0:
        print(f'epoch {epoch + 1}: w={W:.3f}. loss={L:.8f}')
print(f'Prediction after training f(5) = {model(X_test).item():.3f}')


