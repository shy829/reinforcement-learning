import numpy as np
import scipy

# simple solve functions to get v
# v = R + r*Pv'
# c = -2 + 0.9*(10*0.6+0.4*d)
# e = -1 + 0.9*(0.9*e + 0.1*a)
# b = -2 + 0.9*(0.8*c)
# a = -2 + 0.9*(0.5*b + 0.5*e)
# d = 1 + 0.9*(0.2*a+0.4*b+0.4*c)

# A = np.array([[0, 0, 1, -0.36, 0], [-0.09, 0, 0, 0, 0.19], [0, 1, -0.72, 0, 0], [1, -0.45, 0, 0, -0.45],
#               [-0.18, -0.36, -0.36, 1, 0]])
# b = np.array([3.4, -1, -2, -2, 1])
# x = np.linalg.solve(A, b)
# print(x)


# c1,c2,c3,pass,pub,fb,sleep
# v = ((I-r*p)^-1)*R
i = np.identity(7)
p = np.array([[0, 0.5, 0, 0, 0, 0.5, 0], [0, 0, 0.8, 0, 0, 0, 0.2], [0, 0, 0, 0.6, 0.4, 0, 0], [0, 0, 0, 0, 0, 0, 1.0],
              [0.2, 0.4, 0.4, 0, 0, 0, 0],
              [0.1, 0, 0, 0, 0, 0.9, 0], [0, 0, 0, 0, 0, 0, 1]])
r0 = 0.9
r1 = 1
R = np.array([-2, -2, -2, 10, 1, -1, 0])
inv_0 = np.linalg.inv(i - p * r0)
v0 = np.dot(inv_0, R)
print('r=0.9', v0)
print("r=0", np.dot(np.linalg.inv(i - p * 0), R))
print("r=1", np.dot(np.linalg.pinv(i - p), R))
try:
    inv_1 = np.linalg.inv(i - p * r1)
    v1 = np.dot(inv_1, R)
    print(v1)
except Exception as e:
    print(e)

# Markov decision process with action A
# p(a|s) is a distribution over actions given states
# Pss_p = sum(p(a|s)*Pss_a)
# Rs_p = sum(p(a|s)*Rs_a)

# p(a|s) = 0.5, r = 1
# c1,c2,c3,fb,sleep
p_as = 0.5
i = np.identity(5)
p = np.array([[0, 0.5, 0, 0.5, 0], [0, 0, 0.5, 0, 0.5], [0.5 * 0.2, 0.4 * 0.5, 0.4 * 0.5, 0, 0.5],
              [0.5, 0, 0, 0.5, 0], [0, 0, 0, 0, 0]])
R = np.array([-2 * p_as - 1 * p_as, -2 * p_as + 0 * p_as, 10 * p_as + 1 * p_as, -1 * p_as, 0])
r = 1
print("action:", np.dot(np.linalg.pinv(i - p * r), R))
