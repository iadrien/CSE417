# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:24:58 2018

@author: Po Adrich
"""

from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.criterion = "entropy"

#X = [['p','p','r','p','p'],['n','n','y','y','y'],['s','r','s','r','s']]
#X=[[1,1,-1,1,1],[-1,-1,1,1,1],[1,-1,1,-1,1]]
X=[[1,-2,3],[1,-2,-3],[-1,2,3],[1,2,-3],[1,2,3]]
#Y = ['n','n','n','y','y']
Y=[-1,-1,-1,1,1]

clf.fit(X,Y)

dot_data = tree.export_graphviz(clf, out_file='tree.dot') 
