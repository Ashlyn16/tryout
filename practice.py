

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import statsmodels.stats.stattools as stools
from scipy.integrate import odeint
import win32com.client


xl= win32com.client.gencache.EnsureDispatch('Excel.Application')
wb=xl.Workbooks('ExamProblemData2.1.xlsx')
sheet=wb.Sheets('Sheet1')
  
def getdata(sheet, Range):
    data= sheet.Range(Range).Value
    data=scipy.array(data)
    data=data.reshape((1,len(data)))[0]
    return data

x=getdata(sheet,"A2:A11")
ya1=getdata(sheet,"B2:B11")
yp1=getdata(sheet,"C2:C11")
raexp=getdata(sheet,"D2:D11")
rpexp=getdata(sheet,"E2:E11")
'''
print ya1
print raexp
'''
plt.plot(x,ya1,x,yp1)
plt.xlabel('time')
plt.ylabel('conc')

plt.show()

def fit_func(y,k1,k2,n,m):
    y=[ya1,yp1]
    return k1*(y[0]**n)-k2*(y[1]**m)
    
y=[ya1,yp1]    
parameters=curve_fit(fit_func,y,raexp)  
[k1,k2,n,m]=parameters[0]

print k1,k2,n,m


def theor(y,p):
    [ya1,yp1]=y
    [k1,k2,n,m]=p
    rath=k1*(y[0]**n)-k2*(y[1]**m)
    rpth=k2*(y[1]**m)
    return [rath,rpth]
 

def error_func(p,y,rexp):
    [ya1,yp1]=y
    [rath,rpth]=theor(y,p)
    ea=abs(rath-raexp)
    ep=abs(rpth-rpexp)
    err1=ea+ep
    return err1
 
   
rexp=[raexp,rpexp]    
pguess=[-0.05,0.05,0.7,0.3]
plsq=scipy.optimize.leastsq(error_func,pguess,args=(y,rexp))    
p=plsq[0]
rth=theor(y,p)
print rth
print p



'''

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(x,ya1,'ro');
ax.plot(x,yth)   
'''

'''
def deriv(y_th,x,k1,k2,n,m):
    
    [ya_th,yp_th]=y_th
    
    dya_thdx=k1*(ya_th**n)-k2*(yp_th**m)
    dyp_thdx=k2*(yp_th**m)
    return [dya_thdx,dyp_thdx]
 
# initial conditions
ya_th0=ya1[0]
yp_th0=yp1[0]
y_th0=[ya_th0,yp_th0]

# time grid for integration
x=scipy.linspace(0.0,80.0,80.0)


y_th=scipy.integrate.odeint(deriv,y_th0,x)


ya_th=y_th[:,0]
yp_th=y_th[:,1]
ytheor=scipy.array([[ya_th],[yp_th]])

print "theoretical concentrations of reactant and product at time t"
print ytheor
'''









'''
def logistic4(x, A, B, C, D):
    """4PL lgoistic equation."""
    return ((A-D)/(1.0+((x/C)**B))) + D

def residuals(p, y, x):
    """Deviations of data from fitted 4PL curve"""
    A,B,C,D = p
    err = y-logistic4(x, A, B, C, D)  # error=yexp-ycalcul
    return err

def peval(x, p):
    """Evaluated value at x with current parameters."""
    A,B,C,D = p
    return logistic4(x, A, B, C, D)

# Make up some data for fitting and add noise
# In practice, y_meas would be read in from a file
x = np.linspace(0,20,20)
A,B,C,D = 0.5,2.5,8,7.3
y_true = logistic4(x, A, B, C, D)
y_meas = y_true + 0.2*npr.randn(len(x))

# Initial guess for parameters
p0 = [0, 1, 1, 1]

# Fit equation using least squares optimization
plsq = leastsq(residuals, p0, args=(y_meas, x))


'''



















'''
rtheo=k*ya1**n
e1=rexp-rtheo
err1=abs(e1)

print max(err1)
 
'''





'''
k=np.array([0.5,0.35])
#declare the model
def reversible(y,t): 
    [a,p]=y
    dadt= k[0]*a-k[1]*p
    dpdt= k[1]*p
    return dadt,dpdt
    

def solver():
    time =np.linspace(0.0, 80.0, 100.0)
    a0=ya1[0]
    p0=yp1[0]
    y0=[a0,p0] #initaial conc of ya1 and yp1
    y=odeint(reversible,y0,time)
    plt.plot(time,y[:,0],time,y[:,1])
    
    
    f,ax = plt.subplots(5)
    for i in range (0,5):
        ax[i].plot(time, y[:,i])
        ax[i].set_ylabel('y'+str(i))
        
        plt.xlabel('t')
        plt.ylabel('pyode solver')
        
        plt.setp([a.get_xticklabels() for a in f.axes[:]], visible=False)
        plt.setp(ax[4].get_xticklables(), visible=True)
        plt.show()
    
'''    
    

    
'''   

time =np.linspace(0.0, 20.0, 100.0)
yinit=np.array([0.0,0.0])
y=odeint(mymodel,yinit,time)
plt.plot(time,y[:,0],time,y[:,1])
'''

'''
def mymodel(y,t):
    dy=np.zeros(5)
    
    dy[0]= v0 -1/k[0]*y[0]
    dy[1]= k[0]*y[0]-k[1]*y[1]
    dy[2]= v0 -k[2]*y[2]
    dy[3]= v0-k[0]*y[1]+k[2]*y[2]+y[3]
    dy[4]= k[2]*y[1]-k[3]*y[3]-y[4]
    
    return dy
    
def solver():
    
    time=np.linspace(0.0,20.0,100.0)
    yinit=np.zeros(5)
    yinit[4]=1
    y=odeint(mymodel,yinit,time)
    
    f,ax = plt.subplots(5)
    for i in range (0,5):
        ax[i].plot(time, y[:,i])
        ax[i].set_ylabel('y'+str(i))
        
        plt.xlabel('t')
        plt.ylabel('pyode solver')
        
        plt.setp([a.get_xticklabels() for a in f.axes[:]], visible=False)
        plt.setp(ax[4].get_xticklables(), visible=True)
        plt.show()
'''
