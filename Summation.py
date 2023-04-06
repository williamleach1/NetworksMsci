from sympy import *
from sympy.abc import R, l, k, N, z

A = (N-k)/k

f = (l+1)*( ( (N) / (1 + A * z **(-l)) ) )#- ( (N) / (1 + A * z **(-l+1)) ) )

s,e = Sum(f, (l, 1, R-1)).euler_maclaurin(n=5,eps = 1)

print(s)
print(e)