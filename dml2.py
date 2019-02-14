





import numpy as np
import matplotlib.pyplot as plt

def mid_pt(A,B):
	C = (A + B)/2
	return C.T

#perpendicular 

A = np.array([2,3])
B = np.array([4,5])

v0 = np.array([-1,4])

c = 3
v1 = (B-A)
v2 = v0

V = np.vstack((v1,v2))

#ntranspose = (B-A).T

p1 = np.matmul(v1,mid_pt(A,B))
p2 = -c

P = np.vstack((p1,p2))

X = np.matmul(np.linalg.inv(V),P)

#k = B
d = np.linalg.norm(B-X.T)
#e = np.linalg.norm(X-A)

#f = np.sqrt((A[1]-X[1])*(A[1]-X[1]) + (A[0]-X[0])*(A[0]-X[0]))

print(X)
print(d)
#print(f)

x = np.linspace(0,15,10000)
y1 = (x - 3)/4
y2 = X[1] + np.sqrt(d*d - (x-X[0])*(x-X[0]))
y3 = X[1] - np.sqrt(d*d - (x-X[0])*(x-X[0]))

plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)
plt.plot(A[0],A[1],'o')
plt.plot(B[0],B[1],'o')
plt.plot(X[0],X[1],'o')
plt.grid()
plt.show()







