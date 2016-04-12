import scipy
import numpy as npy
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy.integrate

Pma=500*10**-10 # permeability of component A in cm^3(STP)/scm^2cmHg
a=10.0
Pmb=10*Pma # permeability of component B in cm^3(STP)/scm^2cmHg
Qf=1*10**6 # number of moles in feed in cm^3(STP)/s

Pp=19.0 # permeate pressure in cmHg
Pf=190.0 # feed pressure in cmHg
r=Pp/Pf # pressure ratio
xfa=0.21 # mole fraction of oxygen in the feed
L=2.54*10**-3 # lenght of the membrane

  
def deriv(z,Q):
    [l,y]=z
    
    dldQ=(-L*a*z[1]/(1+(a-1)*z[1]))/Pma*Pf*(z[1]-r*(a*z[1]/(1+(a-1)*z[1])))
    dydQ=((a*z[1]/(1+(a-1)*z[1]))-z[1])/Qf
    
    # reference: A simple analysis for gas separation membranes by Richard A. Davis and Orville E. Sandall
    # university of Minnesota Duluth
    return [dldQ,dydQ]

# initial conditions
l0=1000
y0=xfa+0.0001
z0=[l0,y0]
print z0
# time grid for integration
Q=scipy.linspace(1*10**6,100*10**6,11.0)
print Q

p=scipy.integrate.odeint(deriv, z0, Q)

l=p[:,0]
y=p[:,1]
soln=scipy.array([[Q],[l],[y]])

print soln

plt.plot(l,y,'g')
plt.show()




    







