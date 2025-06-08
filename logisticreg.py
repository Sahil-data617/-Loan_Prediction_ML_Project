import pandas as pd 
import numpy as np 
import math
class LogisticRegression:
      def fit(self,x,y):
            self.x=x
            self.y=y
      def ccr(self):
            n=len(self.x)
            r1=(n*np.sum(self.x*self.y))-(np.sum(self.x)*np.sum(self.y))
            r2=math.sqrt(((n*np.sum(self.x**2))-(np.sum(self.x)**2))*((n*np.sum(self.y**2))-(np.sum(self.y)**2)))
            r=r1/r2
            return r
      def slope(self):
            mean_x=np.mean(self.x)
            mean_y=np.mean(self.y)
            xm_mx=self.x-mean_x
            ym_my=self.y-mean_y
            s=(np.sum(xm_mx*ym_my)/np.sum(xm_mx**2))
            return s 
      def intercept(self):
            b1=self.slope()
            b0=np.mean(self.y)-(b1*np.mean(self.x))
            return b0
      def logit(self,input):
            b1=self.slope()
            b0=self.intercept()
            p=math.exp(b0+b1*input)/(1+math.exp(b0+b1*input))
            odd=p/(1-p)
            return odd,p
      


      








df=pd.read_csv("C:/Users/eq5cd/Downloads/credit_score_data.csv")
x=df['CreditScore']
y=df['Approved']
x=np.array(x)
y=np.array(y)
l=LogisticRegression()
l.fit(x,y)
c=l.ccr()
print(c)
k=l.logit(720)
print(k)

