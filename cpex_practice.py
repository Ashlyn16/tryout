
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import statsmodels.stats.stattools as stools
import win32com.client
xl= win32com.client.gencache.EnsureDispatch('Excel.Application')
wb=xl.Workbooks('ExamProblemData2.1.xlsx')
sheet=wb.Sheets('ExamProblemData')
   
def getdata(sheet, Range):
    data= sheet.Range(Range).Value
    data=scipy.array(data)
    data=data.reshape((1,len(data)))[0]
    return data

x=getdata(sheet,"A2:A11")
ya1=getdata(sheet,"B2:B11")
yp1=getdata(sheet,"C2:C11")

'''
y=[ya1,yp1]
time =np.linspace(0.0, 20.0, 100.0)

yinit=np.array([0.0,0.0])
y=odeint(mymodel,yinit,time)

plt.plot(x,y[:,0],x,y[:,1])

plt.xlabel('t')
plt.ylabel('y')
plt.show()
    

f=plt.figure()
plt.subplot(111)
plt.plot(x,y,'g')
plt.show()

plt.subplot(211)
plt.plot(x,yp1,'r')
'''
plt.plot(x,ya1,x,yp1)
plt.xlabel('time')
plt.ylabel('conc')

plt.show()


'''
v0=10
k=np.array([0.5,0.35,0.40,0.21])
#declare the model

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
