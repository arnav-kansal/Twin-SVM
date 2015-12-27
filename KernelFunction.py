import math
import numpy as np

#__copyright__ = ""
#__license__ = "GPL"
# __version__ = "1.1"
# __maintainer__ = "Arnav Kansal"
# __email__ = "ee1130440@ee.iitd.ac.in"
# __status__ = "Production"


def kernelfunction(Type, u, v, p):
    # u, v are array like; p is parameter for kernels; type is:
        # type 1 linear kernel
        # type 2 polynomial kernel
        # type 3 RBF kernels
    if(Type==1):
        return np.dot(u,v)
    if(Type==2):
        return pow(np.dot(u,v)+1,p)
    if(Type==3):
        return pow(math.e,(-np.dot(u-v,u-v)/(p**2)))

def centertrainKernel(K):
    m,n = K.shape
    if(m!=n):
        print("Interrupt!, invalid Kernel")
    else:
        In = np.ones(m,m)/m
        K += np.dot(In,np.dot(K,In)) - (np.dot(K,In)+np.dot(In,K))
        return K

def centertestKernel(K):
    # here K = L*M, L is test data no of points, and M is train data no of points
    l,m = K.shape
    In = np.ones(m,m)/m
    Im = np.ones(l,m)/m
    K += np.dot(Im,np.dot(K,In)) - (np.dot(K,In)+np.dot(Im,K))
    return K