from cmath import isnan
from typing import Tuple
import numpy as np
import math
from scipy.integrate import quad



class distributionStoreUnit():
    def __init__(self):
        self.n = 0
        # self.sum = 0        # the sum of all history data
        # self.squareSum = 0   # the sum of square of all history data
        self.data = []
        self.capacity = 2048
        self.mean = 0       # mean
        self.var = 0        # var
    
    def add(self,newData:float):
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(newData)
        self.mean = np.mean(self.data)
        self.var = np.var(self.data)
        self.n = len(self.data)
        # self.n += 1
        # self.sum += newData
        # self.squareSum += newData**2
        # # update mean and var
        # self.mean = self.sum/self.n
        # self.var = self.squareSum/self.n - self.mean**2
        # if self.n > 0 and self.var==0:
        #     self.var += 1e-3
        # if self.var < 1e-3:
        #     self.var = 1e-3
        # assert self.var>0,"var < 0, var={},newData={}".format(self.var,newData)
        # assert not isnan(self.var),"var is nan, newData={}".format(newData)

    
    def getDist(self):
        return (self.mean,self.var)


def calculateNormalXY(mean:float,var:float):
    sigma = np.sqrt(var)
    X = np.linspace(mean-3*sigma,mean+3*sigma)
    Y = np.exp(-(X-mean)**2/(2*var))/(2.506628275*sigma)
    return X,Y

def calculateOverlap(mean1,var1,mean2,var2,accuracy:int = 1):
    if var1<1e-3:
        var1 = 1e-3
    if var2<1e-3:
        var2 = 1e-3
    sigma1 = np.sqrt(var1)
    sigma2 = np.sqrt(var2)
    if abs(mean1-mean2) > 3*(sigma1+sigma2):    # 如果互相3σ以外没有重叠的 返回0
        return 0
    elif abs(mean1-mean2)<1e-3 and abs(var1-var2)<1e-3:
        return 1.0
    else:
        low_bound = min(mean1-3*sigma1,mean2-3*sigma2)
        up_bound = max(mean1+3*sigma1,mean2+3*sigma2)
        X = np.linspace(low_bound,up_bound,int((up_bound-low_bound)/accuracy))
        y1 = np.exp(-(X-mean1)**2/(2*var1))/(2.506628275*sigma1)
        y2 = np.exp(-(X-mean2)**2/(2*var2))/(2.506628275*sigma2)
        result = 0
        for i in range(len(y1)):
            result += min(y1[i],y2[i])
        assert result>=0,"overlap < 0"
        return result

    
def int_target(x,mean1,sigma1,mean2,sigma2):
    y1 = np.exp(-(x-mean1)**2/(2*sigma1**2))/(2.506628275*sigma1)
    y2 = np.exp(-(x-mean2)**2/(2*sigma2**2))/(2.506628275*sigma2)
    return min(y1,y2)

def calculateOverlap_scipy(mean1,var1,mean2,var2,accuracy:int = 1):
    """
    integral min(normal1,normal2)
    """
    if var1<1e-6:
        var1 = 1e-6
    if var2<1e-6:
        var2 = 1e-6
    sigma1 = np.sqrt(var1)
    sigma2 = np.sqrt(var2)
    low_bound = min(mean1-3*sigma1,mean2-3*sigma2)
    up_bound = max(mean1+3*sigma1,mean2+3*sigma2)
    result = quad(func=int_target,a=low_bound,b=up_bound,args=(mean1,sigma1,mean2,sigma2))
    # print(result)
    return result[0]

if __name__ == '__main__':
    # test code
    test_array = [9,13,8,10,11,9.3,10.4,10.2]
    d = distributionStoreUnit()
    for t in test_array:
        d.add(t)
    print(d.getDist())
    print(np.mean(test_array),np.var(test_array))

    s = calculateOverlap(12,1,15,1)
    print(s)
    s = calculateOverlap_scipy(12,0,12,0.5)
    print(s)