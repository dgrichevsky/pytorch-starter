# npx nodemon .\pytorch.py
import torch
import numpy as np
x = torch.rand(2, 2, dtype=float)
a= torch.ones(5)
b = a.numpy()
a.add_(1)

k = np.ones(5)
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.ones(5, device=device, dtype=float, requires_grad=True)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    # print(z)
    # cannot convert gpu tensor to numpy array
    # z.numpy()
    
# autograd.py
x = torch.randn(3, requires_grad=True)
y = x + 2
print(y) # gradient function calculates dy/dx fn
z=y*y*y*y
print(z)
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
# z = z.mean() #  RuntimeError: grad can be implicitly created only for scalar outputs
z.backward(v) # dz/dx
print(x.grad)

# x.requires_grad_(False) # _ means it modifies variable in place
print(x)
y =x.detach()
print(y)
# with torch.no_grad():
weights = torch.ones(4, requires_grad=True)

# for epoch in range(3):
#     module_output = (weights * 3).sum()
#     module_output.backward()
#     print(weights.grad)
#     weights.grad.zero_()
    
# optimizer = torch.optim.SGD(weights, lr=0.01)
# optimizer.step()
# must empty gradients before next step of optimization
# optimizer.zero_grad()
# print(optimizer)


######## back propagation
"""
1. Forward pass: computes loss
2. compute local gradients
3. backwards pass: computes dLoss / dWeights using the chain rule
"""

x = torch.tensor(1.0) 
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

# step 1
y_hat = w * x
loss = (y_hat - y) ** 2
print(loss)

# backward pass
loss.backward()
print(w.grad)

#update our weights and next forward pass and backward pass
 ############## Gradient Descent algorithm
 #f = w * x
 # f = 2 * x
x = np.array([1, 2,3, 4], dtype=np.float32)
y = np.array([2, 4, 6, 8], dtype=np.float32)
w = 0.0
 # model prediction
def forward(x):
     return w * x
 # loss
def loss(y, y_hat):
     return ((y - y_hat) ** 2).mean()
 # gradient descent
 # MSE = 1 / N * (w* x -y  ) ** 2
# dJ / dW = 1/ N 2x (w*x -y)
def gradient(x, y,y_hat):
    return np.dot(2 * x, y_hat - y).mean()

print(f'Prediction before training f(5) = {forward(5):.3f}')
learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(x)
    #loss
    l = loss(y, y_pred)
    # gradients
    dw = gradient(x, y, y_pred)
    
    # update weights
    w -= learning_rate * dw
    
    if epoch % 1 == 0:
        print(f'epoch {epoch + 1}: w={w:.3f}. loss={l:.8f}')
print(f'Prediction after training f(5) = {forward(5):.3f}')


#### using torch here
A = torch.tensor([1, 2,3, 4], dtype=torch.float32)
B = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
W = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
def forwardTorch(x):
     return W * x


for epoch in range(n_iters):
    # prediction = forward pass
    B_pred = forwardTorch(A)
    #loss
    L = loss(B, B_pred)
    # gradients = backward pass
    L.backward() # dL/dW
    
    # update weights
    with torch.no_grad():
        W -= learning_rate * W.grad
    W.grad.zero_()
    
    if epoch % 1 == 0:
        print(f'epoch {epoch + 1}: w={W:.3f}. loss={L:.8f}')
print(f'Prediction after training f(5) = {forwardTorch(5):.3f}')


 