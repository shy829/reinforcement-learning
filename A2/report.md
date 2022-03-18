## RL Assignment2

### Requirements

Programming language: python3

You should implement both first-visit and every-visit MC method and TD(0) to evaluate an uniform random policy π(n|·) = π(e|·) = π(s|·) = π(w|·) = 0.25.



### My Implementation

#### MC_first_visit

```
V(St) = V(St) + 1/N(St)(Gt-V(St))
V is the state value
Gt is the final reward
N is the number of times
define try_times by oneself
for first_visit: calculate the average of the first access to s
```

use a new list to store the first access to s

```
If we use set() to delete repeat contents, we can't make sure that the remained number in set is the first one appears in the formal list
```



#### MC_every_visit

similar to first_visit and calculate everything



#### TD0

```
V(S) = V(S) + alpha[R+gamma*V(S')-V(S)]
V(S') is the value of next state
alpha is step size defined by oneself
gamma is discount factor
others are similar to mc
```



### Result

#### 10000 times

<img src="/Users/sunheyu/Library/Application Support/typora-user-images/image-20220317191535014.png" alt="image-20220317191535014" style="zoom:67%;" />

mc_first_visit:
-54.29 -43.40 -62.04 -71.32 -75.92 -81.05 
-63.70 -61.10 -66.06 -70.13 -73.41 -77.41 
-70.96 -68.91 -69.56 -70.20 -71.78 -74.86 
-76.74 -72.86 -71.78 -70.89 -69.90 -70.74 
-81.00 -76.03 -73.13 -70.55 -66.41 -62.40 
-86.67 -80.62 -76.50 -72.14 -62.52 -43.31 
mc_every_visit:
-60.02 -42.48 -71.72 -88.53 -96.07 -98.88 
-76.26 -75.25 -84.07 -92.65 -97.29 -99.58 
-89.53 -91.76 -95.68 -98.18 -97.83 -96.41 
-100.33 -100.48 -100.62 -99.42 -95.69 -91.08 
-105.93 -104.64 -102.27 -97.09 -86.47 -75.84 
-108.85 -106.58 -102.63 -93.11 -74.24 -43.79 
td0_learning:
-3.56 0.00 -5.45 -7.56 -8.92 -9.11 
-5.37 -5.03 -6.96 -8.35 -8.83 -8.87 
-7.91 -7.72 -8.20 -8.67 -8.69 -8.67 
-8.69 -8.57 -8.63 -8.58 -8.12 -7.90 
-9.18 -9.01 -8.95 -8.30 -7.29 -6.02 
-9.33 -9.19 -8.93 -8.20 -5.22 0.00 

Comparing to the result in the first assignment, we can see that all three methods fit well.

The first_visit performs better in convergence than every_visit, and the every_visit has a larger Value state since it takes every step.

TD0 performs best in convergence. (Comparing to 100000 times)

#### **100000 times**

`first try`

mc_first_visit:
-53.70 -42.60 -61.67 -70.04 -75.18 -80.26 
-63.45 -60.00 -64.95 -68.82 -72.22 -76.51 
-70.77 -67.84 -68.57 -69.59 -70.95 -73.84 
-75.44 -71.81 -70.66 -69.93 -68.79 -70.05 
-79.10 -74.74 -72.11 -69.35 -65.06 -61.77 
-84.13 -79.22 -75.87 -70.89 -61.80 -42.70 
mc_every_visit:
-59.61 -42.68 -73.89 -90.18 -98.10 -101.78 
-76.87 -75.50 -85.50 -93.60 -98.57 -100.52 
-91.16 -91.47 -94.09 -96.61 -97.46 -97.24 
-100.63 -100.24 -98.95 -97.04 -93.31 -89.53 
-106.24 -104.57 -100.74 -94.60 -85.07 -73.65 
-108.26 -106.21 -101.15 -91.78 -74.87 -42.70 
td0_learning:
-3.25 0.00 -4.43 -7.35 -8.66 -9.05 
-5.04 -5.06 -6.27 -7.96 -8.73 -8.96 
-7.55 -7.96 -7.71 -8.43 -8.63 -8.65 
-8.83 -8.89 -8.64 -8.48 -8.25 -8.11 
-9.15 -9.12 -8.95 -8.36 -7.19 -4.85 
-9.29 -9.17 -8.68 -7.75 -5.97 0.00 

`second try`

mc_first_visit:
-54.82 -42.87 -62.28 -70.91 -75.78 -80.79 
-64.80 -61.00 -65.85 -69.62 -72.65 -77.06 
-71.79 -68.40 -69.37 -70.24 -71.51 -74.86 
-76.39 -72.43 -71.37 -70.45 -69.50 -70.76 
-79.61 -75.34 -72.82 -69.93 -65.84 -62.19 
-84.51 -79.89 -76.45 -71.91 -62.87 -43.18 
mc_every_visit:
-59.59 -42.42 -73.46 -89.91 -97.79 -100.84 
-76.05 -74.36 -84.32 -92.89 -97.78 -100.17 
-89.81 -90.38 -93.51 -95.91 -96.87 -97.14 
-99.40 -98.85 -98.37 -96.07 -92.01 -89.00 
-105.14 -103.66 -100.22 -93.77 -84.20 -73.38 
-107.60 -105.34 -100.51 -91.04 -74.25 -42.75 
td0_learning:
-3.36 0.00 -4.07 -7.02 -8.35 -9.09 
-5.50 -6.17 -7.42 -8.18 -8.71 -9.06 
-7.87 -7.44 -8.42 -8.80 -8.90 -8.80 
-8.83 -8.83 -8.92 -8.87 -8.43 -7.99 
-9.30 -9.21 -8.93 -8.47 -7.45 -6.77 
-9.43 -9.37 -8.97 -8.13 -4.30 0.00 

#### 1000000 times

mc_first_visit:
-53.75 -42.44 -61.62 -70.15 -75.22 -80.32 
-63.64 -60.30 -65.07 -69.09 -72.30 -76.63 
-70.91 -67.84 -68.75 -69.89 -71.12 -74.23 
-75.59 -71.92 -70.91 -70.06 -69.15 -70.30 
-79.27 -74.91 -72.42 -69.60 -65.52 -61.88 
-84.08 -79.35 -75.90 -71.17 -62.37 -42.87 
mc_every_visit:
-58.88 -42.34 -73.67 -89.56 -97.58 -100.84 
-75.81 -74.38 -84.61 -93.09 -97.95 -99.95 
-90.10 -90.28 -93.33 -96.03 -96.88 -96.74 
-99.42 -98.92 -98.24 -96.35 -92.72 -88.97 
-105.02 -103.52 -100.17 -94.07 -84.53 -73.47 
-107.43 -105.28 -100.33 -91.05 -74.28 -42.86 
td0_learning:
-4.50 0.00 -6.52 -7.86 -9.02 -9.31 
-6.26 -4.64 -6.74 -8.12 -9.06 -9.16 
-7.86 -7.48 -8.42 -8.67 -8.88 -8.60 
-8.70 -8.68 -8.79 -8.69 -8.18 -7.23 
-9.12 -9.20 -8.91 -8.34 -7.43 -4.06 
-9.39 -9.24 -8.79 -8.39 -4.98 0.00 



100000 times is already time-costing, and 1000000 times  even cost minutes. But it seems good in result.



### Summary and Thinking

TD0 performs better than MC methods.

In TD0, by setting different discount factors, we get different results. The closer it is to 1, the value state at the terminal state is larger. Step size is also important, setting it too large may cause it overfitting, while tiny step size make it slower to reach convergence state.

