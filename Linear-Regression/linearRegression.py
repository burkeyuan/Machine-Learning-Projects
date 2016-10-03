import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(fileName):
    data=np.loadtxt(fileName)
    xVal=data[:,0:2]
    yVal=data[:,2]
    plt.plot(xVal[:,1],yVal,'ro')
    plt.title("Data From "+fileName)
    plt.xlabel("xVal")
    plt.ylabel("yVal")
    plt.show()
    return xVal,yVal


def standRegres(xVal,yVal):
    x_mtx=np.matrix(xVal)
    y_mtx=np.matrix(yVal)
    theta=(x_mtx.getT()*x_mtx).getI()*x_mtx.getT()*y_mtx.getT()
    plt.plot(x_mtx[:,1],yVal,'ro',x_mtx[:,1],theta[0]+np.multiply(theta[1],x_mtx[:,1]),'b-',linewidth=2.0)
    plt.xlabel("xVal")
    plt.ylabel("yVal")
    plt.title("Linear Regression(NE)")
    plt.show()
    return theta


def polyRegres(xVal,yVal):
    x_mtx=np.zeros((xVal.shape[0],3))
    x_mtx[:,0]=1
    x_mtx[:,1]=xVal[:,1]
    x_mtx[:,2]=xVal[:,1]*xVal[:,1]
    y_mtx=np.matrix(yVal)
    theta=np.polyfit(xVal[:,1],yVal,3)
    t=np.arange(min(xVal[:,1]),max(xVal[:,1]),(max(xVal[:,1])-min(xVal[:,1]))/200)
    plt.plot(xVal[:,1],yVal,'ro',t,theta[3]+theta[2]*t+theta[1]*t*t+theta[0]*t*t*t,'b-',linewidth=2.0)
    plt.xlabel("xVal")
    plt.ylabel("yVal")
    plt.title("Polynomial Regression(NE)")
    plt.show()
    return theta


if __name__ == '__main__':
    xVal,yVal=loadDataSet('Q2data.txt')
    theta1=standRegres(xVal,yVal)
    print(theta1)
    theta2=polyRegres(xVal,yVal)
    print(theta2)