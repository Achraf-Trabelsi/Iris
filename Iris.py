
from datetime import date
import pandas
from sklearn import datasets
import matplotlib
import pylab as pl
from itertools import cycle
from sklearn import naive_bayes
import numpy as np
import random as r
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

irisData = datasets.load_iris()
#modele naive bayes 
nb = naive_bayes.MultinomialNB(fit_prior=True) # un algo d'apprentissage

nb.fit(irisData.data[:], irisData.target[:])
p31 = nb.predict(irisData.data[31].reshape(1,-1))
#print(p31)
plast = nb.predict(irisData.data[-1].reshape(1,-1))
#print(plast)
p = nb.predict(irisData.data[:])
#print(p)
print('Boucle for:')
#l'erreur avec une boucle
ea = 0
for i in range(len(irisData.data)):
 if (p[i] != irisData.target[i]):
  ea = ea+1

print('Erreur: ',ea/len(irisData.data))
a1=1 -ea/len(irisData.data)
print('Accuracy: ',a1)

#l'erreur d'un seule coup
tab=p-irisData.target
#print(tab)
#print(type(tab))
print('\nNon_zero')
print('Erreur: ',np.count_nonzero(tab)/len(irisData.data))
a=nb.score(irisData.data,irisData.target)

print('\nScore function')
print('Erreur :',1-a)
# print(type(irisData.data))
# print(irisData.data.shape)
# print(irisData.target.shape)
#fonction de departage
def split(S):

 dataS1=np.empty(shape=(135,4))
 targetS1=np.empty(shape=(135))
 dataS2=S.data
 targetS2=S.target
 size =int(0.9*len(S.data))
 for i in range(0,size):
      j=r.randint(0,len(dataS2)-1)
      dataS1[i]= dataS2[j]
      targetS1[i]=targetS2[j]
      dataS2=np.delete(dataS2,j,0)
      targetS2=np.delete(targetS2,j,0) 
 
#  print([dataS1,targetS1,dataS2,targetS2])
#  print(targetS1.shape)
#  print(dataS1.shape)
#  print(dataS2.shape)
#  print(targetS2.shape)
 return([dataS1,targetS1,dataS2,targetS2])

split(irisData)
def test(S,clf):
  [train,target1,test,target2]=split(S)
  clf.fit(train[:], target1[:])
  p = clf.predict(test[:])
  a=clf.score(test,target2)
  #print(a)
  return(1-a)

#Naive Bayes Multinomial
clf = naive_bayes.MultinomialNB(fit_prior=True)
test(irisData, clf)
t=10
#print('\nErreur avec Naive Bayes')
err=[]
err_mean_tab_nb =[]
for j in range(0,20):
  for i in range(0,t):
    e=test(irisData,clf)
    err.append(e)
  err_mean=sum(err,0)/len(err)
  err_mean_tab_nb.append(err_mean)
  #print(j,' ',err_mean)

#Decision Tree
clf1 = DecisionTreeClassifier()
test(irisData, clf)

#print('\nErreur avec Decision Tree')
err=[]
err_mean_tab_dt =[]
for j in range(0,20):
  for i in range(0,t):
    e=test(irisData,clf)
    err.append(e)
  err_mean=sum(err,0)/len(err)
  err_mean_tab_dt.append(err_mean)
  #print(j,' ',err_mean)

tab = np.empty(shape=(21,2))
tab[:-1,0] = err_mean_tab_nb
tab[:-1,1] = err_mean_tab_dt
tab[-1,0] = np.mean(err_mean_tab_nb)
tab[-1,1] = np.mean(err_mean_tab_dt)

columns = ['Naive Bayes', 'Decision Tree' ]
rows =[]
for i in range(0,20) :
  rows.append(i)

rows.append('Mean')

df = pandas.DataFrame(tab, columns=columns, index= rows)
print('\n',df)