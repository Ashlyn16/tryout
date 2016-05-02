import scipy
import scipy.integrate
import matplotlib.pyplot as plt

def deriv(Y,x,a,b):
    [y,z]=Y
    dydx=z
    dzdx=a*y+b*scipy.exp(x)
    return [dydx,dzdx]
y0,z0=1.0,2.0
Y0=[y0,z0]
x=scipy.linspace(0.0,1.0,10.0)    
a=4.0
b=1.0

soln=scipy.integrate.odeint(deriv,Y0,x,args=(a,b))
y=soln[:,0]
z=soln[:,1]
A=scipy.array([[x],[y],[z]])


print A

plt.plot(x,y,'g')
plt.show()



'''
    
    def f(Jw):
        f=(math.exp(Jw/Kl)/(R+(1-R)*math.exp(Jw/Kl)))-(Cm/Cb)
        return f
    Jw=fsolve(f,0.0001) # cm/s
    Cb1.append(Cb)
    Jw1.append(Jw)
    print(Jw1)
f=plt.figure()
plt.subplot(111)
plt.plot(Cb1,Jw1,'g')
plt.xlabel('Cb1')
plt.ylabel('Jw1')

plt.show()

print (Jw1)             
print "the concentration polarization is given as",(Cm/Cb)
print "solute concentration at membrane surface is",(Cm)
print "salt flux is",(Js)
print "thickness of mass transfer film is",(d);
'''

    
