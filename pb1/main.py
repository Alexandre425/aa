import numpy as np
import matplotlib.pyplot as plt

a = np.array([[1,2,3,4],[5,6,7,8]]) 
print(a)        # Array
print(a[1,2])   # A specific element
print(a[0,:])   # A line: Slice notation
print(a[:,0])   # A column
print(a[0,0:2]) # A subset of a line [start idx : end number]

print(a[0,0:4:2])   # A subset with a stepsize of 2
print(a[0,0:-1])    # Excluding the last element

z = np.zeros((2,3))
o = np.ones((2,3))
f = np.full((3,2), 100)     # Fills the tensor with 100
fl = np.full_like(f, 4)     # Fills a tensor with the same shape as f with 4
r = np.random.rand(4,2)     # Random 4x2 tensor with rand numbers between 0 and 1
ri = np.random.randint(-4, 8, size=(3,3))    # Random ints between -4 and 8
ident = np.identity(5)      # 5x5 identity matrix

m = np.matmul(o,f)          # Matrix multiplication
d = np.linalg.det(m)        # Matrix determinant
mi = np.min(r, axis=0)      # Line with the minimum values
re = np.reshape(f, (2,3))   # Reshaping a tensor 
t = f.T                     # Transpose a tensor    THESE ARE NOT THE SAME!
print(t)

a = np.array([1,2,3,4,5,6], dtype='int32')
b = np.ones((6), dtype='int32')
plt.plot(a, a**2, 'rx')
plt.show()