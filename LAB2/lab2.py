import numpy as np
import random , math
from scipy . optimize import minimize 
import matplotlib . pyplot as plt

# parameters is a datapoint (in this case di=(xi,yi))
# linear kernel
def linear_kernel(di,dj):
    ker = np.dot(di,dj)
    return ker

# Polynomial kernel
p = 8
def poly_kernel(di,dj):
    ker = (np.dot(di,dj) + 1)**p
    return ker

# Radial kernel
sigma = 6
def rad_kernel(di,dj):
    sq_dist = np.linalg.norm(di-dj)**2
    ker = math.exp(-sq_dist/(2*sigma**2))
    return ker

ker = poly_kernel   # Choosing Kernel 
C = None         # Choosing slack coeff, C = None if no slack

# Import data
nA = 10             # Number of samples for class A
nB = 20             # Number of samples for class B
np.random.seed(100)
classA = np.concatenate( 
    (np.random.randn(nA, 2) * 0.2 + [1.5, 0.5],
    np.random.randn(nA, 2) * 0.2 + [-1.5, 0.5])) 

classB = np.random.randn(nB, 2) * 0.2 + [0.0 , 0.5]
inputs = np. concatenate(( classA , classB )) 
targets = np.concatenate(
    (np.ones(classA.shape[0]) , 
    -np.ones(classB.shape[0])))
N = inputs.shape[0] # Number of rows (samples)
permute=list(range(N)) 
random.shuffle(permute) 
inputs = inputs[ permute , : ]
targets = targets[ permute ]


# Construct P matrix including the factor 0.5
P = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        P[i][j] = 0.5*targets[i]*targets[j]*ker(inputs[i],inputs[j])
    
# Define objective function alphaT*P*alpha-the sum of alpha
def objective(alpha):
    res = np.dot(alpha,np.dot(P,alpha)) - np.sum(alpha)
    return res


# Define zerofunc sum of alphai*ti = alpha•target
def zerofun(alpha):
    res = np.dot(alpha,targets)
    return res


alpha0 = np.zeros(N)                # Starting value for alpha
B = [(0,C) for b in range(N)]       # Conditions for alphas, C = None => no slack
XC = {'type':'eq', 'fun':zerofun}   # Equality constraint 

ret = minimize( objective , alpha0 ,
                bounds=B, constraints=XC )
alpha = ret['x']
success = ret['success']
print('Found solution = {}'.format(success))
tol = 1e-5      # Tolerance for floating point error
# Separate and store non neative alphas with corresponing input and target
filtered_data = [ [value, inputs[idx], targets[idx]] for idx, value in enumerate(alpha) if value > tol]

# Find marginal points and select one of them (no condition of which)
# marginal_point is a list of alpha with corresponing input and target

if C == None:       # If no slack
    marginal_points = [point for point in filtered_data]
    margin_point = marginal_points[0]
    
else:               # If slack one need to choose alpha < C
    marginal_points = [point for point in filtered_data if point[0] < C-tol] # Added tolerance for float error
    margin_point = marginal_points[0]         
print('Marginal points: {}'.format(len(marginal_points)))

# Caluclate the sum of alphai*ti*K(s,xi)
def sum_sv(data, s):
    S = 0
    for point in data:
        alph = point[0]
        x = point[1]
        t = point[2]
        S += alph*t*ker(s,x)
    return S

# Selects xi and ti from the arbitrarily chosen marginal point
xs = margin_point[1]
ts = margin_point[2]

# Calcuates b
b = sum_sv(filtered_data,xs) - ts  

# Indicator function which determines the target of new inputs s = (x,y)
def indicator(s):
    ind = sum_sv(filtered_data,s) - b
    return ind

# Plot the samples
plt.plot([p[0] for p in classA],
        [p[1] for p in classA],
        'b.')

plt.plot([p[0] for p in classB],
        [p[1] for p in classB],
        'r.')

# Plot the contour lines
xgrid = np.linspace(-5,5)
ygrid = np.linspace(-8,4)

grid = np.array([[indicator([x,y]) 
                for x in xgrid] 
                for y in ygrid])

plt.contour(xgrid,ygrid,grid, 
            (-1.0,0.0,1.0), 
            colors=('red','black','blue'),
            linewidths=(1,3,1))

# Plots the marginal points if solution found
if success:
    plt.plot([m[1][0] for m in marginal_points],
            [m[1][1] for m in marginal_points]
            ,'g*')

# Legends
plt.legend(['Class A', 'Class B','Marginal points'])

plt.axis('equal')
plt.savefig('svmplot.png')
plt.show()


