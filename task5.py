from numpy import*
#F1+F2+F3= 1000kg/h total mass balance
#0.08F1+ 0F2+ 0.02F3= 30 SALT BALANCE
# 0F1+ 0.15F2+ 0.04F3= 50 ETHANOL BALANCE
A = array([[1,1,1],
           [0.08,0,0.02],
           [0,0.15,0.04 ]],dtype=float)
b = array([1000,30,50], dtype=float)
D=diag(A)
R=A-diag(D)
x0=zeros(3, float)
TOL=1*e-4; MAXIT=200
diag_dom=True
for i in range(3):
    sum=0
    for j in range(3):
        if i!=j:
            sum+=abs(A[i][j])
    if abs(A[i][i]) < abs(sum):
        diag_dom=False
        print("it doesn't strictly diagonally dominant")
    else:
        print("it strictly diagonally dominant")

#JACOBI METHOD
for k in range(1,MAXIT+1):
    x_new=(b-R@x0)/D
    if linalg.norm(x_new-x0)/linalg.norm(x_new)<TOL:
        x0=x_new
        break
x0=x_new
residual=linalg.norm(b-A@x0)
print("jacobi")
print("x=", x0)
print("iteration=", k)
print("residual=", residual)
#GAUSS-SEIDEL METHOD

A=array([[1,1,1],
          [0.08,0,0.02],
          [0,0.15,0.04]],dtype=float)
b= array([1000,30,50], dtype=float)
D=diag(A)
L=tril(A,-1)
U=triu(A,1)
x0=zeros(3)
TOL=1*e-4; MAXIT=200
DL_inv=linalg.inv(D+L)
for k in range(1,MAXIT+1):
    x_new=DL_inv@(b-U@x0)
    if linalg.norm(x_new-x0)/linalg.norm(x_new)<TOL:
        x0=x_new
        break
residual=linalg.norm(b-A@x0)
print("gauss-seidel")
print("x=", x0)
print("iteration=", k)
print("residual=", residual)





