import scipy
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt

#import matplotlib.pyplot as plt`


l=10.0
w=1.0
a=2.0
R=0.95
P=100*10^(-10)
dl=0.5
Qfi=200.0
cfi=0.8
cp_bar=0.5
Qp=115.0


while dl<l:
    dQpdl=P*w*(cfi*((1-Qp/Qfi)**(-R))-cp_bar)/((l*a*cfi*((1-Qp/Qfi)**(-R)))/(1+(a-1)*cfi*((1-Qp/Qfi)**(-R))))
    Qp=Qp+dQpdl*dl
    Qf=Qfi-Qp
    cf=cfi*((1-Qp/Qfi)**(-R))
    dl=dl+0.5
    cfi=cf
    Qfi=Qf
    print (Qfi, dl)
    plt.plot(dl,Qfi,'r')
    plt.xlabel('distance')
    plt.ylabel('retentate flow rate')
plt.show()
    
    
    
    
    
'''    
list_of_Qfi=[Qfi]    
list_of_dl=[]
for Qfi in list_of_Qfi:
    st1.Qfi=Qfi
    dl=st1.dl()
    print (Qfi, dl)
    plt.plot(dl,Qfi,'r')
    plt.xlabel('distance')
    plt.ylabel('retentate flow rate')
    plt.show()'''
    
print "the retentate concentration is", (cfi)    
print "the retentate flowrate is", (Qfi)


    