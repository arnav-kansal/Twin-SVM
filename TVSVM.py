import numpy as np
from sklearn import preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
import KernelFunction as kf
import TwinPlane1
import TwinPlane2

#__copyright__ = ""
#__license__ = "GPL"
# __version__ = "1.1"
# __maintainer__ = "Arnav Kansal"
# __email__ = "ee1130440@ee.iitd.ac.in"
# __status__ = "Production"

class TwinSVMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,Epsilon1=0.1, Epsilon2=0.1, C1=1, C2=1,kernel_type=0,kernel_param=1,regulz1=1, regulz2=1,fuzzy=0,_estimator_type="classifier"):
        self.Epsilon1=Epsilon1
        self.Epsilon2=Epsilon2
        self.C1=C1
        self.C2=C2
        self.regulz1 = regulz1
        self.regulz2 = regulz2
        self.fuzzy = fuzzy
        self.kernel_type=kernel_type
        self.kernel_param=kernel_param
        
    def fit(self, X, Y):
        assert (type(self.Epsilon1) in [float,int])
        assert (type(self.Epsilon2) in [float,int])
        assert (type(self.C1) in [float,int])
        assert (type(self.C2) in [float,int])
        assert (type(self.regulz1) in [float,int])
        assert (type(self.regulz2) in [float,int])
        assert (self.fuzzy in [0,1])
        assert (type(self.kernel_param) in [float,int])
        assert (self.kernel_type in [0,1,2,3])
        ####################fill in here
        # mean -centering, doing std 
        #X_t = preprocessing.scale(X)
        # Data Sorting, rearranging
        Data = sorted(zip(Y,X), key=lambda pair: pair[0], reverse = True)
        Total_Data = np.array([np.array(x) for y,x in Data])
        A=np.array([np.array(x) for y,x in Data if (y==1)])
        B=np.array([np.array(x) for y,x in Data if (y==0)])
        # Radius, center of data calcs
        if(self.fuzzy==1):
            if(self.kernel_type==0):
                rcenpos=0
                rcenneg=0
                xcenpos = np.true_divide(sum(A),len(A))
                for a in A:
                    if(rcenpos<np.linalg.norm(a-xcenpos)):
                        rcenpos = np.linalg.norm(a-xcenpos)
                xcenneg = np.true_divide(sum(B),len(B))
                for b in B:
                    if(rcenneg<np.linalg.norm(b-xcenneg)):
                        rcenneg = np.linalg.norm(b-xcenneg)
                self.xcenpos_ = xcenpos
                self.xcenneg_ = xcenneg
                self.rcenpos_ = rcenpos
                self.rcenneg_ = rcenneg
            else:
                rcenpossq=-np.inf
                termtemp1=0
                for i in range(len(A)):
                    term1 = kf.kernelfunction(self.kernel_type,A[i],A[i],self.kernel_param)
                    term2 = 0
                    for j in range(len(A)):
                        term2 += kf.kernelfunction(self.kernel_type,A[j],A[i],self.kernel_param)
                        termtemp1 += kf.kernelfunction(self.kernel_type,A[i],A[j],self.kernel_param)
                    term2 = -2*term2/len(A)
                    rcenpossq = max(rcenpossq,term1+term2)
                termtemp1 = termtemp1/(len(A)*len(A))
                rcenpossq += termtemp1
                rcennegsq=-np.inf
                termtemp2=0
                for i in range(len(B)):
                    term1 = kf.kernelfunction(self.kernel_type,B[i],B[i],self.kernel_param)
                    term2 = 0
                    for j in range(len(B)):
                        term2 += kf.kernelfunction(self.kernel_type,B[j],B[i],self.kernel_param)
                        termtemp2 += kf.kernelfunction(self.kernel_type,B[i],B[j],self.kernel_param)
                    term2 = -2*term2/len(B)
                    rcennegsq = max(rcennegsq,term1+term2)
                termtemp2 = termtemp2/(len(B)*len(B))
                rcennegsq += termtemp2
                self.rcenpossq_ = rcenpossq
                self.rcennegsq_ = rcennegsq
                self.termtemp1_ = termtemp1
                self.termtemp2_ = termtemp2
        m1 = A.shape[0]
        m2 = B.shape[0]
        e1 = -np.ones((m1,1))
        e2 = -np.ones((m2,1))
        if(self.kernel_type==0): # no need to cal kernel
            S = np.hstack((A,-e1))
            R = np.hstack((B,-e2))
        else:
            S = np.zeros((A.shape[0],Total_Data.shape[0]))
            for i in range(A.shape[0]):
                for j in range(Total_Data.shape[0]):
                    S[i][j] = kf.kernelfunction(self.kernel_type,A[i],Total_Data[j],self.kernel_param)
            S = np.hstack((S,-e1))
            R = np.zeros((B.shape[0],Total_Data.shape[0]))
            for i in range(B.shape[0]):
                for j in range(Total_Data.shape[0]):
                    R[i][j] = kf.kernelfunction(self.kernel_type,B[i],Total_Data[j],self.kernel_param)
            R = np.hstack((R,-e2))
        #####################Calculation of Function Parameters(Equation of planes) 
        [w1,b1] = TwinPlane1.Twin_plane_1(R,S,self.C1,self.Epsilon1,self.regulz1)
        [w2,b2] = TwinPlane2.Twin_plane_2(S,R,self.C2,self.Epsilon2,self.regulz2)
        self.plane1_coeff_ = w1
        self.plane1_offset_ = b1
        self.plane2_coeff_ = w2
        self.plane2_offset_ = b2
        self.data_ = Total_Data
        self.A_ = A
        self.B_ = B
        return self


    def get_params(self, deep=True):
        return {"Epsilon1": self.Epsilon1, "Epsilon2": self.Epsilon2, "C1": self.C1, "C2": self.C2, "regulz1": self.regulz1,
                "regulz2":self.regulz2, "kernel_type": self.kernel_type, "kernel_param": self.kernel_param,"fuzzy": self.fuzzy}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

    def predict(self, X, y=None):
        #X_test = preprocessing.scale(X)    
        if(self.kernel_type==0): # no need to cal kernel
            S = X
            w1mod = np.linalg.norm(self.plane1_coeff_)
            w2mod = np.linalg.norm(self.plane2_coeff_)
        else:
            S = np.zeros((self.data_.shape[0],self.data_.shape[0]))
            for i in range(self.data_.shape[0]):
                for j in range(self.data_.shape[0]):
                    S[i][j] = kf.kernelfunction(self.kernel_type,self.data_[i],self.data_[j],self.kernel_param)
            w1mod = np.sqrt(np.dot(np.dot(self.plane1_coeff_.T,S),self.plane1_coeff_))
            w2mod = np.sqrt(np.dot(np.dot(self.plane2_coeff_.T,S),self.plane2_coeff_))
            S = np.zeros((X.shape[0],self.data_.shape[0]))
            for i in range(X.shape[0]):
                for j in range(self.data_.shape[0]):
                    S[i][j] = kf.kernelfunction(self.kernel_type,X[i],self.data_[j],self.kernel_param)
        y1 = np.dot(S,self.plane1_coeff_)+ ((self.plane1_offset_)*(np.ones((X.shape[0],1))))
        y2 = np.dot(S,self.plane2_coeff_)+ ((self.plane2_offset_)*(np.ones((X.shape[0],1))))

        ###############Compute test data predictions
        yPredicted=np.zeros((X.shape[0],1))

        distFromPlane1 = y1/w1mod #abs(np.dot(Z.transpose(),self.plane1_coeff_)+self.plane1_offset_)
        distFromPlane2 = y2/w2mod #abs(np.dot(Z.transpose(),self.plane2_coeff_)+self.plane2_offset_)

        for i in range(len(distFromPlane1)):
            if (distFromPlane1[i]<distFromPlane2[i]):
                yPredicted[i][0]=0;
            else:
                yPredicted[i][0]=1;

        return yPredicted.transpose()[0]    

    def decision_function(self,X):
        #X_test = preprocessing.scale(X)    
# membership function:
#   1-(x+ - xi)/r+
#   1-(x_ - xi)/r_
#kernel
#   1-_/(|di^2|/(r+^2))
#   1-_/(|di^2|/(r-^2))    
        #fuzzy=0
        if(self.fuzzy==1):
            s1=[]
            s2=[]  
            if(self.kernel_type==0): # no need to cal kernel
                for i in range(len(X)):
                    s1.append(1-(np.linalg.norm(self.xcenpos_-X[i])/self.rcenpos_))
                    s2.append(1-(np.linalg.norm(self.xcenneg_-X[i])/self.rcenneg_))
            else:
                for i in range(len(X)):
                    dsquaredpos = kf.kernelfunction(self.kernel_type,X[i],X[i],self.kernel_param)
                    term1 = 0
                    for j in range(len(self.A_)):
                        term1 += kf.kernelfunction(self.kernel_type,self.A_[j],X[i],self.kernel_param)
                    term1 = -2*term1/len(self.A_)
                    dsquaredpos += term1
                    dsquaredpos += self.termtemp1_
                    s1.append(1-np.sqrt(dsquaredpos/self.rcenpossq_))
                    dsquaredneg = kf.kernelfunction(self.kernel_type,X[i],X[i],self.kernel_param)
                    term1 = 0
                    for j in range(len(self.B_)):
                        term1 += kf.kernelfunction(self.kernel_type,self.B_[j],X[i],self.kernel_param)
                    term1 = -2*term1/len(self.B_)
                    dsquaredneg += term1
                    dsquaredneg += self.termtemp2_
                    s2.append(1-np.sqrt(dsquaredneg/self.rcennegsq_))
            s1 = np.array(s1)
            s2 = np.array(s2)
            return np.true_divide(s1,s1+s2)-0.5
        else:
            if(self.kernel_type==0): # no need to cal kernel
                S = X
                w1mod = np.linalg.norm(self.plane1_coeff_)
                w2mod = np.linalg.norm(self.plane2_coeff_)
            else:
                S = np.zeros((self.data_.shape[0],self.data_.shape[0]))
                for i in range(self.data_.shape[0]):
                    for j in range(self.data_.shape[0]):
                        S[i][j] = kf.kernelfunction(self.kernel_type,self.data_[i],self.data_[j],self.kernel_param)
                w1mod = np.sqrt(np.dot(np.dot(self.plane1_coeff_.T,S),self.plane1_coeff_))
                w2mod = np.sqrt(np.dot(np.dot(self.plane2_coeff_.T,S),self.plane2_coeff_))
                S = np.zeros((X.shape[0],self.data_.shape[0]))
                for i in range(X.shape[0]):
                    for j in range(self.data_.shape[0]):
                        S[i][j] = kf.kernelfunction(self.kernel_type,X[i],self.data_[j],self.kernel_param)
            y1 = np.dot(S,self.plane1_coeff_)+ ((self.plane1_offset_)*(np.ones((X.shape[0],1))))
            y2 = np.dot(S,self.plane2_coeff_)+ ((self.plane2_offset_)*(np.ones((X.shape[0],1))))

        ###############Compute test data predictions
            yPredicted=np.zeros((X.shape[0],1))

            distFromPlane1 = y1/w1mod 
            distFromPlane2 = y2/w2mod 
        ###############Compute test data predictions
       
            for i in range(len(distFromPlane1)):
                yPredicted[i][0] = distFromPlane2[i]/(distFromPlane1[i]+distFromPlane2[i])-0.5
            return yPredicted.transpose()[0]