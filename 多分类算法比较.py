# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 20:36:09 2023

@author: hp
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import BernoulliNB


data=pd.read_csv(r'C:\Users\hp\Desktop\CD.csv',encoding='gb2312')
data.head()

x=data.iloc[:,1:6].values
y=data.iloc[:, 0].values 
test_size=0.4
seed=88
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=seed,stratify=y)
sc=StandardScaler()
sc.fit(x_train)
x_train=sc.transform(x_train)
x_test=sc.transform(x_test)

#lda=LDA(n_components=2)
#lda=lda.fit(x_train,y_train)

clf=DecisionTreeClassifier(random_state=1
                          ,max_depth=6
                          ,min_samples_leaf=2
                          ,criterion='entropy'
                          )
rfc=RandomForestClassifier(random_state=10
                          ,max_depth=3
                          ,min_samples_leaf=1
                          ,criterion='entropy'
                          ,n_estimators=100)

clf=clf.fit(x_train,y_train)
rfc=rfc.fit(x_train,y_train)
score_c=clf.score(x_test,y_test)
score_r=rfc.score(x_test,y_test)
print("SingleTree:{}".format(score_c)
      ,"RandomForest:{}".format(score_r))

for kernel in ["poly"]:
    svr=SVC(kernel=kernel
            #,gamma="auto"
            #,degree=1
            #,cache_size=5000
           ).fit(x_train,y_train)
    result=svr.predict(x_test)
    score=svr.score(x_test,y_test)
#LR
LR=LogisticRegression(random_state=0)
LR.fit(x_train,y_train)
LogisticRegression(random_state=0)
#KNN
KNN = KNeighborsClassifier(n_neighbors=6, algorithm = 'ball_tree')
KNN = KNN.fit(x_train, y_train)
#Naive Bayes
BNB = BernoulliNB()
BNB = BNB.fit(x_train,y_train)

gpc = GaussianProcessClassifier(random_state=0)
gpc = gpc.fit(x_train, y_train)

score_BNB1 = cross_val_score(BNB,x_train,y_train,cv=2)
BNB_train = score_BNB1.mean()
BNBtrain=np.std(score_BNB1 ,ddof = 1)
#Naive Bayes_test
score_BNB2 = cross_val_score(BNB,x_test,y_test,cv=3)
BNB_test = score_BNB2.mean()
BNBtest=np.std(score_BNB2 ,ddof = 1)
print("BNB_train:{}".format(BNB_train)
      ,"BNB_test:{}".format(BNB_test))
#GCP_train
score_gpc1 = cross_val_score(gpc,x_train,y_train,cv=3)
gpc_train = score_gpc1.mean()
gpctrain=np.std(score_gpc1 ,ddof = 1)
score_gpc2 = cross_val_score(gpc,x_test,y_test,cv=3)
gpc_test = score_gpc2.mean()
gpctest=np.std(score_gpc2 ,ddof = 1)
print("gpc_train:{}".format(gpc_train)
      ,"gpc_test:{}".format(gpc_test))
#KNN_train
score_KNN1 = cross_val_score(KNN,x_train,y_train,cv=4)
KNN_train = score_KNN1.mean()
KNNtrain=np.std(score_KNN1 ,ddof = 1)
#KNN_test
score_KNN2 = cross_val_score(KNN,x_test,y_test,cv=2)
KNN_test = score_KNN2.mean()
KNNtest=np.std(score_KNN2 ,ddof = 1)
print("KNN_train:{}".format(KNN_train)
      ,"KNN_test:{}".format(KNN_test))
#lda_train
#score_lda1 = cross_val_score(lda,x_train,y_train,cv=3)
#lda_train = score_lda1.mean()
#ldatrain=np.std(score_lda1 ,ddof = 1)
#score_lda2 = cross_val_score(lda,x_test,y_test,cv=3)
#lda_test = score_lda2.mean()
#ldatest=np.std(score_lda2 ,ddof = 1)
#print("lDA_train:{}".format(lda_train)
#      ,"LDA_test:{}".format(lda_test))

#RF_train
score_rfc1 = cross_val_score(rfc,x_train,y_train,cv=6)
rfc_train = score_rfc1.mean()
np.std(score_rfc1 ,ddof = 1)
#RF_test
score_rfc2 = cross_val_score(rfc,x_test,y_test,cv=4)
rfc_test = score_rfc2.mean()
np.std(score_rfc2 ,ddof = 1)
print("RF_train:{}".format(rfc_train)
      ,"RF_test:{}".format(rfc_test))
#DT_train
score_clf1 = cross_val_score(clf,x_train,y_train,cv=3)
clf_train = score_clf1.mean()
Clftrain=np.std(score_clf1 ,ddof = 1)
#DT_test
score_clf2 = cross_val_score(clf,x_test,y_test,cv=3)
clf_test = score_clf2.mean()
Clftest=np.std(score_clf2 ,ddof = 1)
print("DT_train:{}".format(clf_train)
      ,"DT_test:{}".format(clf_test))
#SVM_train
score_svm1 = cross_val_score(svr,x_train,y_train,cv=3)
svm_train = score_svm1.mean()
svmtrain=np.std(score_svm1 ,ddof = 1)
score_svm2 = cross_val_score(svr,x_test,y_test,cv=3)
svm_test = score_svm2.mean()
svmtest=np.std(score_svm2 ,ddof = 1)
print("svm_train:{}".format(svm_train)
      ,"svm_test:{}".format(svm_test))
#LR_train
score_LR1 = cross_val_score(LR,x_train,y_train,cv=3)
LR_train = score_LR1.mean()
LRtrain=np.std(score_LR1 ,ddof = 1)
#LR_test
score_LR2 = cross_val_score(LR,x_test,y_test,cv=3)
LR_test = score_LR2.mean()
LRtest=np.std(score_LR2 ,ddof = 1)
print("LR_train:{}".format(LR_train)
      ,"LR_test:{}".format(LR_test))

