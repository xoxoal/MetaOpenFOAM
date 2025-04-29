import sys
sys.path.append('/data/Chenyx/MetaOpenFOAM3/active_subspaces/active_subspaces')
sys.path.append('/data/Chenyx/MetaOpenFOAM3/active_subspaces/active_subspaces/utils')

import active_subspaces as ac
import numpy as np
import matplotlib.pyplot as plt
import numpy as np     
from utils.response_surfaces import PolynomialApproximation
from scipy.stats import qmc


def wing(xx):
    #each row of xx should be [Sw. Wfw, A, Lambda, q, lambda, tc, Nz, Wdg, Wp] in the normalized input space
    #returns column vector of wing function at each row of inputs
    
    x = xx.copy()
    x = np.atleast_2d(x)
    M = x.shape[0]
    
    Sw = x[:,0]; Wfw = x[:,1]; A = x[:,2]; L = x[:,3]*np.pi/180.; q = x[:,4]
    l = x[:,5]; tc = x[:,6]; Nz = x[:,7]; Wdg = x[:,8]; Wp = x[:,9]
    
    return (.036*Sw**.758*Wfw**.0035*A**.6*np.cos(L)**-.9*q**.006*l**.04*100**-.3*tc**-.3*Nz**.49*Wdg**.49 + Sw*Wp).reshape(M, 1)
    
def wing_grad(xx):
    #each row of xx should be [Sw. Wfw, A, Lambda, q, lambda, tc, Nz, Wdg, Wp] in the normalized input space
    #returns matrix whose ith row is gradient of wing function at ith row of inputs
    
    x = xx.copy()
    x = np.atleast_2d(x)
    
    Sw = x[:,0]; Wfw = x[:,1]; A = x[:,2]; L = x[:,3]*np.pi/180.; q = x[:,4]
    l = x[:,5]; tc = x[:,6]; Nz = x[:,7]; Wdg = x[:,8]; Wp = x[:,9]
    
    Q = .036*Sw**.758*Wfw**.0035*A**.6*np.cos(L)**-.9*q**.006*l**.04*100**-.3*tc**-.3*Nz**.49*Wdg**.49 #Convenience variable
    
    dfdSw = (.758*Q/Sw + Wp)[:,None]
    dfdWfw = (.0035*Q/Wfw)[:,None]
    dfdA = (.6*Q/A)[:,None]
    dfdL = (.9*Q*np.sin(L)/np.cos(L))[:,None]
    dfdq = (.006*Q/q)[:,None]
    dfdl = (.04*Q/l)[:,None]
    dfdtc = (-.3*Q/tc)[:,None]
    dfdNz = (.49*Q/Nz)[:,None]
    dfdWdg = (.49*Q/Wdg)[:,None]
    dfdWp = (Sw)[:,None]
        
    return np.hstack((dfdSw, dfdWfw, dfdA, dfdL, dfdq, dfdl, dfdtc, dfdNz, dfdWdg, dfdWp))


M = 1000 #This is the number of data points to use

#Sample the input space according to the distributions in the table above
Sw = np.random.uniform(150, 200, (M, 1))
Wfw = np.random.uniform(220, 300, (M, 1))
A = np.random.uniform(6, 10, (M, 1))
L = np.random.uniform(-10, 10, (M, 1))
q = np.random.uniform(16, 45, (M, 1))
l = np.random.uniform(.5, 1, (M, 1))
tc = np.random.uniform(.08, .18, (M, 1))
Nz = np.random.uniform(2.5, 6, (M, 1))
Wdg = np.random.uniform(1700, 2500, (M, 1))
Wp = np.random.uniform(.025, .08, (M, 1))

#The input matrix
x = np.hstack((Sw, Wfw, A, L, q, l, tc, Nz, Wdg, Wp))

#The function's output
f = wing(x)

#Upper and lower limits for inputs
ub = np.array([150, 220, 6, -10, 16, .5, .08, 2.5, 1700, .025]).reshape((1, 10))
lb = np.array([200, 300, 10, 10, 45, 1, .18, 6, 2500, .08]).reshape((1, 10))

#We normalize the inputs to the interval [-1, 1]: 
XX = 2.*(x - lb)/(ub - lb) - 1.0


lb = [150, 220, 6, -10, 16, .5, .08, 2.5, 1700, .025]
ub = [200, 300, 10, 10, 45, 1, .18, 6, 2500, .08]
def latin_hypercube_sampling(N, lb, ub, num_samples):
    """
    拉丁超立方取样
    
    参数:
    N : int
        变量数量（维度数）
    lb : numpy.ndarray
        每个变量的下界，大小为 (N,)
    ub : numpy.ndarray
        每个变量的上界，大小为 (N,)
    num_samples : int
        取样点的数量
    
    返回:
    samples : numpy.ndarray
        生成的拉丁超立方样本，大小为 (num_samples, N)
    """
    # 创建拉丁超立方采样器
    sampler = qmc.LatinHypercube(d=N)
    
    # 生成 [0, 1] 区间的拉丁超立方样本
    samples = sampler.random(n=num_samples)
    
    # 将样本从 [0, 1] 区间映射到 [lb, ub] 区间
    scaled_samples = qmc.scale(samples, lb, ub)
    
    return scaled_samples

# 示例：生成5个变量的拉丁超立方样本
N = 10
num_samples = 1000  # 样本数量

samples = latin_hypercube_sampling(N, lb,ub,num_samples)

print('samples:', samples[0])
#Instantiate a subspace object
ss = ac.subspaces.Subspaces()

#Compute the subspace with a global linear model (sstype='OLS') and 100 bootstrap replicates
ss.compute(X=XX, f=f, nboot=100, sstype='OLS')

print("eigenvecs",ss.eigenvecs[0].reshape(10, 1))

ac.utils.plotters.eigenvectors(ss.eigenvecs[0].reshape(10, 1))
#This plots the eigenvalues (ss.eigenvals) with bootstrap ranges (ss.e_br)
ac.utils.plotters.eigenvalues(ss.eigenvals, ss.e_br)

#This plots subspace errors with bootstrap ranges (all contained in ss.sub_br)
ac.utils.plotters.subspace_errors(ss.sub_br)

#This makes sufficient summary plots with the active variables (XX.dot(ss.W1)) and output (f)
ac.utils.plotters.sufficient_summary(XX.dot(ss.W1), f)

#quadratic polynomial approximation
RS = PolynomialApproximation(2)

#Train the surface with active variable values (y = XX.dot(ss.W1)) and function values (f)
y = XX.dot(ss.W1)
RS.train(y, f)
print ('The R^2 value of the response surface is {:.4f}'.format(RS.Rsqr))

#Plot the data and response surface prediction
plt.figure(figsize=(7, 7))
y0 = np.linspace(-2, 2, 200)
plt.plot(y, f, 'bo', y0, RS.predict(y0[:,None])[0], 'k-', linewidth=2, markersize=8)
plt.grid(True)
plt.xlabel('Active Variable Value', fontsize=18)
plt.ylabel('Output', fontsize=18)
figname = 'figs/response_surface.png'
plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)