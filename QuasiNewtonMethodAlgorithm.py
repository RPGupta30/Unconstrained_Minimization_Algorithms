#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''Unconstrained Optimization Algorithm'''
'''Quasi Newton Method Algorithm'''

# Please change the function value as per our requirement and also change it's derivative

import numpy as np
from numpy import linalg as LA 
import matplotlib.pyplot as plt 
import sympy
from sympy.utilities.lambdify import lambdify


# Extracting main function
def f_x(f_expression, values):
    f = lambdify((v[0],v[1]), f_expression)                     
    return f(values[0],values[1])                               

#Extract gradients
def df_x(f_expression, values):
    df1_sympy = np.array([sympy.diff(f_expression, i) for i in v])       #first order derivatives
    dfx_0 = lambdify((v[0],v[1]), df1_sympy[0])                          #derivative w.r.t x_0
    dfx_1 = lambdify((v[0],v[1]), df1_sympy[1])                          #derivative w.r.t x_1
    evx_0 = dfx_0(values[0], values[1])                                  #evaluating the gradient at given values
    evx_1 = dfx_1(values[0], values[1])
    return(np.array([evx_0,evx_1]))

#Extract Hessian
def hessian(f_expression):
    df1_sympy = np.array([sympy.diff(f_expression, i) for i in v])              #first order derivatives
    hessian = np.array([sympy.diff(df1_sympy, i) for i in v]).astype(np.float)  #hessian
    return(hessian)

# Plotting the selected function
def loss_surface(sympy_function):
    return(sympy.plotting.plot3d(sympy_function, adaptive=False, nb_of_points=400))



#Function to create a countour plot
def contour(sympy_function):
    x = np.linspace(-3, 3, 100)                         
    y = np.linspace(-3, 3, 100)                         
    x, y = np.meshgrid(x, y)                            
    func = f_x(sympy_function, np.array([x,y]))
    return plt.contour(x, y, func)

#Function to plot contour along with the travel path of the algorithm
def contour_travel(x_array, sympy_function):
    x = np.linspace(-2, 2, 100)                         #x-axis
    y = np.linspace(-2, 2, 100)                         #y-axis
    x, y = np.meshgrid(x, y)                            #creating a grid using x & y
    func = f_x(sympy_function, np.array([x,y]))
    plt.contour(x, y, func)
    plot = plt.plot(x_array[:,0],x_array[:,1],'x-')
    return (plot)




v = sympy.Matrix(sympy.symbols('x[0] x[1]'))

#creating functions for use
# Here we I have choosen thee functions to evaluate our algorithms


f_sympy1 = v[0]**2 - 2.0 * v[0] * v[1] + 4 * v[1]**2        
f_sympy2 = 0.5*v[0]**2 + 2.5*v[1]**2                           
f_sympy3 = 4*v[0]**2 + 2*v[1]**2 + 4*v[0]*v[1] - 3*v[0]  

# take any one function
f = f_sympy3

# Extracting the function
f = f_x(f, v)
print('The selected function is: ',f)
print()

# Finding the gradient of the function
df = df_x(f, v)
print('Gradient of the function is: ',df)
print()

# Finding the hessian of the function
hess_f = hessian(f)
print('Hessian of the function is: ',hess_f)
print('')


# Visualizing the selected function
fun = loss_surface(f)

# Plotting the contour
con = contour(f)



def QuasiNewton(sympy_function, max_iter, start, step_size , epsilon ):
    i = 0
    x_values = np.zeros((max_iter+1,2))
    x_values[0] = start
    norm_values = []
    
    B = np.zeros((max_iter+1,2,2))
    a = np.random.random_integers(-20,20+1,size=(2,2))
    b = a.T
    B[0] = (a+b)/2
    
    
    while i < max_iter:
        norm = LA.norm(df_x(sympy_function, x_values[i]))
        if norm < epsilon:
            break
        else:
            B_inv = LA.inv(B[i])
            grad = df_x(sympy_function, x_values[i])
            #print(grad)
            p = -np.dot(grad, B_inv)
            #print(p)
            x_values[i+1] = x_values[i] + step_size*p
            
            del_k = x_values[i+1] - x_values[i]
            gamma_k = np.dot(del_k, grad)
            
            
            
            B[i+1] = B[i] + np.dot(gamma_k.T,gamma_k)/np.dot(del_k,gamma_k.T) - np.dot(np.dot(gamma_k,del_k.T),np.dot(del_k,gamma_k))/np.dot(np.dot(del_k,gamma_k),del_k.T)
            
            
            
            norm_values.append(norm)
        i+=1
    print("No. of steps Quasi Newton's method took to converge: ", len(norm_values))
    print('')
    return(x_values, norm_values)

x_QNM, norm_values = QuasiNewton(sympy_function=f, max_iter=10, start =[0,0] , step_size = 1, epsilon = 10**-2)
print(x_QNM)

print()
print("contour path through Quasi Newton's method")
Contour_path = contour_travel(x_QNM, f)

