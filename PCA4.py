# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:04:40 2017

@author: hasee
"""

import numpy as np
from sklearn import linear_model
import pandas as pd
import os
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.decomposition import PCA
import statsmodels.tsa.stattools as ts
from sklearn.mixture import GMM
from sklearn.cluster import DBSCAN
import seaborn as sns
import math

font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 27} #90

matplotlib.rc('font', **font)

#from heatmap2 import heatmap
delete = 1
os.chdir(r'C:\Users\wys\Downloads\mirna analysis\mirna analysis')
if delete==1:
    upper = 11    
    df = pd.read_csv('matrix2.csv')
    day_array = [0,1,3,4,7,8,10,14,17] 
    day_label = ['P0','P1','P3','P4','P7','P8','P10','P14','P21']
elif delete==0:
    upper = 15
    df = pd.read_csv('matrix.csv')
    day_array = [0,1,2,3,4,5,7,8,10,12,14,17,21]    
    day_label = ['P0','P1','P2','P3','P4','P5','P7','P8','P10','P12','P14','P17','P21']
color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
marker = ['.',',','o','v','^','<','>','1','2','3','4','8','s','p','*','h','H','+','*','D','d','|','_']
#求出各个miRNA的均值，分成均值大于3和小于3的两组
#df.iloc[:,2:upper] = np.log2(df.iloc[:,2:upper] + 1)
df_mean = pd.concat([df.iloc[:,0:2],df.iloc[:,2:upper].mean(axis=1)],axis=1)
df_mean.rename({'0':'mean'},inplace = True)
df_mean_f = df
df_mean_f2 = df[df_mean[0]!=0]

#df_mean_f.iloc[:,2:16] = np.log2(df_mean_f.iloc[:,2:16]+1)
#画出所有miRNA的表达量，其中小于3的用蓝色标注

#x = np.arange(0,14)

#实验验证的miRNA
df_valid = pd.DataFrame([['10138','mmu-miR-130a-3p','1','0.972','0.419','0.456','0.714','0.754','0.45','0.572','0.221'],
                         ['10972','mmu-miR-181b-5p','1','1.09','0.474','0.808','1.52','1.595','0.752','1.729','0.7'],
                         ['30687','mmu-miR-93-5p','1','1.331','0.715','0.8','0.724','0.377','0.215','0.114','0.027'],
                         ['4040','mmu-miR-9-5p','1','2.894','1.596','1.541','1.272','1.875','1.377','1.478','0.629'],
                         ['29852','mmu-miR-9-3p','1','1.4','1.297','1.083','2.682','4.506','2.79','1.829','3.386'],
                         ['42865','mmu-miR-181a-5p','1','0.976','0.612','0.856','2.353','1.392','1.458','0.865','1.688'],
                         ['14328','mmu-miR-124-3p','1','1.131','0.935','2.212','3.709','5.679','11.127','33.933','43.472'],
                         ['145845','mmu-miR-20a-5p','1','0.686','0.626','0.573','0.434','0.277','0.107','0.034','0.028'],
                         ['169075','mmu-miR-92a-3p','1','1.05','0.9','0.67','0.79','0.35','0.24','0.35','0.3'],
                         ['42619','mmu-miR-709','1','0.483','0.317','0.722','0.777','1.192','1.167','2.442','1.831']],
                        columns=['ID','Name','P0','P1','P3','P4','P7','P8','P10','P14','P21'])
                        # ['146201','mmu-miR-1839-3p','1','0.956','1.317','1.011','1.768','1.331','0.683','1.991','2.801']
def demonstration4(sort=None, section='line', log=False): #所有miRNA的表达量

    x = np.array(day_array).reshape(-1,1)
    y = 0
    c = 0
    mean = pd.DataFrame((df_mean_f.iloc[:,2:upper].mean(axis=1)).T,columns = ['mean'])
    data = df_mean_f
    if log == True:
        data.iloc[:,2:upper]+= .00001
        data.iloc[:,2:upper] = np.log10(data.iloc[:,2:upper])

    if sort!=None:
        data = pd.concat([data, mean],axis = 1)
        data = data.sort_values(by = 'mean')
        '''
        c = 0
        for i in data.ix[:,'mean']:
            if i>sort:
                break
            c+=1
        data = data.iloc[c:,:]
        '''
        data = data.drop('mean',1)

    if section=='line':
        fig = plt.figure(figsize=(24,24))    
        ax = Axes3D(fig)
        for i in range(len(data)):
            z1 = data.iloc[i,2:upper]
            ax.plot(x,z1,y)        
            y+=1
        ax.view_init(90, -90)
        
    elif section=='heat':
        fig, ax = plt.subplots(figsize=(12, 24))
        c = ax.pcolor(data.iloc[:,2:upper],cmap = 'RdBu_r')
        fig.colorbar(c, ax=ax)
        ax.set_xticklabels(day_label,
                           minor = False, fontsize = 7)
        
    ax.set_xlabel('time')
    ax.set_ylabel('miRNAs')
l = ['mmu-miR-9-5p', 'mmu-miR-130a-3p', 'mmu-miR-181a-5p', 'mmu-miR-181b-5p', 'mmu-miR-92a-3p', 'mmu-miR-20a-5p', 'mmu-miR-93-5p', 'mmu-miR-9-3p', 'mmu-miR-124-3p', 'mmu-miR-378a-3p', 'mmu-miR-129-1-3p', 'mmu-miR-125a-5p','mmu-miR-709','mmu-miR-1839-3p']
        
#'mmu-miR-9-3p', 'mmu-miR-129-1-3p', 'mmu-miR-101a-3p', 'mmu-miR-93-5p','mmu-miR-20a-5p','mmu-miR-130a-3p','mmu-miR-92a-3p','mmu-miR-16-5p'       
#'mmu-miR-9-3p','mmu-miR-124-3p','mmu-miR-96-5p','mmu-miR-129-1-3p','mmu-miR-5100','mmu-miR-101a-3p','mmu-miR-3963','mmu-miR-181a-5p','mmu-miR-9-5p','mmu-miR-93-5p','mmu-miR-20a-5p','mmu-miR-130a-3p','mmu-miR-92a-3p','mmu-miR-125a-5p','mmu-miR-16-5p'
#'mmu-miR-9-3p','mmu-miR-124-3p','mmu-miR-96-5p','mmu-miR-5100','mmu-miR-3963','mmu-miR-709','mmu-miR-129-1-3p','mmu-miR-181a-5p','mmu-miR-9-5p','mmu-miR-130a-3p','mmu-miR-181b-5p'
def demonstration6(names): #画对应mirna的热图
    data = df_mean_f
    l = pd.DataFrame()
    for i in names:
        temp = data[data['Name'] == i]
        l = pd.concat([l,temp],axis = 0)
    print(l)
    for i in range(l.shape[0]):
        l.iloc[i,2:upper] = l.iloc[i,2:upper] - l.iloc[i, 2]
    sns.set(rc = {"figure.figsize": (63, 36)}, font_scale=5)
    ax = sns.heatmap(l.iloc[:,2:upper], yticklabels = names, cmap="YlGnBu")

    return ax

def demonstration11(): #融合了3和6
    fig,(ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(24,12))
    l = ['mmu-miR-9-5p', 'mmu-miR-130a-3p', 'mmu-miR-181a-5p', 'mmu-miR-181b-5p',
         'mmu-miR-92a-3p', 'mmu-miR-20a-5p', 'mmu-miR-93-5p', 'mmu-miR-9-3p', 
         'mmu-miR-124-3p', 'mmu-miR-378a-3p', 'mmu-miR-129-1-3p', 'mmu-miR-125a-5p',
         'mmu-miR-709','mmu-miR-1839-3p']
    ax1 = demonstration6(l)
    ax2 = demonstration3('both', threshold=0.829)
    
#求miRNA的k,r2值，保存到df_k_f中
df_k_f=df_mean_f.iloc[:,0:2]
df_k_f['k'] = 0
df_k_f['r'] = 0

regr = linear_model.LinearRegression()
#x = x.reshape(-1,1)
x = np.array(day_array).reshape(-1,1)
a_list=[]
r_list = []
for i in range(len(df_mean_f)):
    z2 = df_mean_f.iloc[i,2:upper].reshape(-1,1)
    regr.fit(x,z2)
    a,b =regr.coef_,regr.intercept_
    r_square=regr.score(x,z2)
    a_list.append(a)
    r_list.append(r_square)
a_list = np.array(a_list).reshape(-1,1)
r_list = np.array(r_list).reshape(-1,1)
df_k_f['k'] = a_list
df_k_f['r'] = r_list
for i in range(len(df_k_f)):
    if df_k_f.ix[i,'k']==0: df_k_f.ix[i,'r'] = 0
#将miRNA的k值，标准化保存到df_k_f中

#将miRNA的k值分为向上增加和向下减少两类

df_k_pos = df_k_f[df_k_f['k']>=0]
df_k_neg = df_k_f[df_k_f['k']<=0]
df_mean_f_pos = df_mean_f[df_k_f['k']>=0]
df_mean_f_neg = df_mean_f[df_k_f['k']<=0]
df_mean_f_pos = df_mean_f_pos.reset_index(drop=True)
df_mean_f_neg = df_mean_f_neg.reset_index(drop=True)
df_k_pos['std_k'] = 0
df_k_neg['std_k'] = 0

std_k_list = []
kmin = np.min(df_k_pos.iloc[:,2])
kmax = np.max(df_k_pos.iloc[:,2])
for i in range(len(df_k_pos)):
    std_k_list.append((df_k_pos.iloc[i,2] - kmin)/(kmax-kmin))
std_k_list = np.array(std_k_list).reshape(-1,1)
df_k_pos['std_k'] = std_k_list

std_k_list = []
kmin = np.min(np.abs(df_k_neg.iloc[:,2]))
kmax = np.max(np.abs(df_k_neg.iloc[:,2]))
for i in range(len(df_k_neg)):
    std_k_list.append(np.abs((df_k_neg.iloc[i,2] - kmin))/(kmax-kmin))
std_k_list = np.array(std_k_list).reshape(-1,1)
df_k_neg['std_k'] = std_k_list

df_k_f['std_k'] = 0
std_k_list = []
kmin = np.min(np.abs(df_k_f.iloc[:,2]))
kmax = np.max(np.abs(df_k_f.iloc[:,2]))
for i in range(len(df_k_f)):
    std_k_list.append((np.abs(df_k_f.iloc[i,2]) - kmin)/(kmax-kmin))
std_k_list = np.array(std_k_list).reshape(-1,1)
df_k_f['std_k'] = std_k_list    

df_k_neg = df_k_neg.reset_index(drop=True)
df_k_pos = df_k_pos.reset_index(drop=True)

#画出miRNA上调和下调的表达图
y = 0
fig2 = plt.figure(figsize=(24,24))
ax2 = Axes3D(fig2)
for i in range(len(df_mean_f_pos)):
    z1 = df_mean_f_pos.iloc[i,2:upper]
    ax2.plot(x,z1,y)
    y+=1
ax2.view_init(90, -90)

y = 0
fig4 = plt.figure(figsize=(24,24))
ax4 = Axes3D(fig4)
for i in range(len(df_mean_f_neg)):
    z1 = df_mean_f_neg.iloc[i,2:upper]
    ax2.plot(x,z1,y)
    y+=1
ax4.view_init(90, -90)

'''
estimator = KMeans(n_clusters=3) #构造聚类器
estimator.fit(df_mean_f.iloc[:,2:16]) #聚类
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
inertia = estimator.inertia_ # 获取聚类准则的总和
'''


p = 5
#PCA降至2维
#pca = PCA(n_components=2)

#data = pd.DataFrame(pca.fit_transform(df_mean_f_neg.iloc[:,2:16]))
data = df_mean_f_pos.iloc[:,2:upper]
#k均值聚类

[centroid, label, inertia] = cluster.k_means(data,p)

[centroid_pos, label_pos, inertia_pos] = cluster.k_means(df_mean_f[df_k_f['k']>=0].iloc[:,2:upper],3)

[centroid_neg, label_neg, inertia_neg] = cluster.k_means(df_mean_f[df_k_f['k']<=0].iloc[:,2:upper],3)

#gmm聚类
'''
gmm = GMM(n_components = p).fit(data)
label = gmm.predict(data)
'''

#DBSCAN
'''
label = DBSCAN(eps=0.3, min_samples=4).fit_predict(data)
'''

label = pd.DataFrame(label)
fig = plt.figure(figsize=(24,24))

#x = np.arange(0,14)
x = np.array(day_array).reshape(-1,1)

def demonstration5(k, absolute = True, slope='k',seg = None): #斜率直方图，'k'为实际值，'std_k'为相对值
    if k=='pos': data = df_k_pos
    if k=='neg': data = df_k_neg
    if k=='both': data = df_k_f
    fig, ax =plt.subplots(figsize=(18,16))
    data.ix['s']=0
    for i in range(len(data)):
        if data.ix[i,'k']>=0:
            data.ix[i,'s']=1
        else:
            data.ix[i,'s']=-1
    if absolute==True:data.ix[:,slope] = np.abs(data.ix[:,slope])
    ax.hist(data.ix[:,slope],bins = np.arange(0,4,0.05))
    ax.set_xlabel('Absolute value of miRNAs')
    ax.set_ylabel('Frequency')
    if seg!=None:
        ax.axvline(x=seg, color ='black',linewidth = 1)
        ax.annotate('dividing line', xy = (1.5,0.1+seg), xytext=(1.5,0.1+seg),arrowprops=dict(facecolor='black', shrink=0.05))
        
def demonstration7(absolute = True, slope='k',seg = None): #斜率直方图，'k'为实际值，'std_k'为相对值
    data_pos = df_k_pos
    data_neg = df_k_neg
    fig, (ax1,ax2) =plt.subplots(nrows=1, ncols=2, figsize=(18,12))
    data_neg.ix[:,slope] = np.abs(data_neg.ix[:,slope])
    ax1.hist(data_pos.ix[:,slope],bins = np.arange(0,4,0.05))
    ax1.set_xlabel('Absolute value of up-regulated miRNAs')
    ax1.set_ylabel('Frequency')    
    ax2.hist(data_neg.ix[:,slope],bins = np.arange(0,4,0.05))
    ax2.set_xlabel('Absolute value of down-regulated miRNAs')
    ax2.set_ylabel('Frequency')           

'''
for k in range(1,p+1):
    ax5 = fig.add_subplot(4, 2, k, projection='3d')
    y = 0
    df_temp = df_mean_f_neg[label[0]==k-1]
    df_temp = df_temp.reset_index(drop=True)
    for i in range(len(df_temp)):
        z1 = df_temp.iloc[i,2:16]
        ax5.plot(x,z1,y)
        y+=1
    ax5.view_init(90, -90)

'''

''' 以std_k和r为向量做PCA
pca = PCA(n_components=1)

data = pca.fit_transform(df_k_pos.iloc[:,3:5])
df_k_f['pca'] = 0
df_k_f['pca'] = pd.DataFrame(data)

data = pca.fit_transform(df_k_pos.iloc[:,3:5])
df_k_pos['pca'] = 0
df_k_pos['pca'] = pd.DataFrame(data)

data = pca.fit_transform(df_k_neg.iloc[:,3:5])
df_k_neg['pca'] = 0
df_k_neg['pca'] = pd.DataFrame(data)
'''
def demonstration3(k,threshold=None, s='k', annotation=0, xlim=None, seg=None): #s为斜率或标准化斜率
    #以std_k为横轴，R2为纵轴画散点图
    #threshold=0.829
    global df_k_f,df_mean_f,df_k_pos,df_k_neg
    fig3, ax3 = plt.subplots(figsize=(42, 42))
    #ax3 = matplotlib.axes.Axes

    df_k_f = df_k_f.reset_index(drop=True) #刷新索引号
    df_mean_f = df_mean_f.reset_index(drop=True)
    if k=='pos': data = df_k_pos
    if k=='neg': data = df_k_neg
    if k=='both': data = df_k_f
    data = data.reset_index(drop=True)
    data.ix[:,s] = np.abs(data.ix[:,s])
    n = 0
    for i in range(len(data)):
        #ax3.scatter(df_k_neg.ix[i,'k'], df_k_neg.ix[i,'r'], color = color[label.ix[i,0]], marker = marker[label.ix[i,0]])
        #ax3.scatter(abs(data.ix[i,'k']), data.ix[i,'r'])
        #if annotation==0: ax3.annotate(data.ix[i,'Name'], xy = (data.ix[i,'k'], data.ix[i,'r']))
        if xlim!=None: ax3.set_xlim([0,xlim])
        if threshold!=None:
            if data.ix[i,s]>=threshold: 
                ax3.scatter(data.ix[i,s], data.ix[i,'r'], marker='x', 
                            s=600, color='red', linewidth=10)
                ax3.annotate(data.ix[i,'Name'], xy = (data.ix[i, s], data.ix[i, 'r']), fontsize=30)   
                n+=1
        else: 
            ax3.scatter(data.ix[i,s], data.ix[i,'r'], linewidth=10)
            ax3.annotate(data.ix[i,'Name'], xy = (data.ix[i,s], data.ix[i,'r']))
    if seg!=None:
        ax3.axhline(y = seg, color ='black',linewidth = 1)
        #ax3.annotate('dividing line', xy = (seg,900), xytext=(seg+0.5, 900),arrowprops=dict(facecolor='black', shrink=0.05))
    ax3.set_xlabel('Absolute value of slope', labelpad=60, fontsize=60)
    ax3.set_ylabel('Coefficient of determination', labelpad=60, fontsize=60)
    ax3.tick_params(labelsize=40, width=10)
    ax3.set_ylim([0.6,1])
    return ax3
        #ax3.annotate(df_k_neg.ix[i,'Name'], xy = (df_k_neg.ix[i,'std_k'], df_k_neg.ix[i,'r']))
        
def demonstration8(threshold_pos = None, threshold_neg = None):
    fig,(ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18,6))
    data_pos = df_k_pos
    data_neg = df_k_neg
    data_neg.ix[:,'k'] = np.abs(data_neg.ix[:,'k'])
    n = 0
    for i in range(len(data_pos)):
        if threshold_pos!=None:
            if data_pos.ix[i,'k']>=threshold_pos: 
                ax1.scatter(data_pos.ix[i,'k'], data_pos.ix[i,'r'], color = 'b')
                ax1.annotate(data_pos.ix[i,'Name'], xy = (data_pos.ix[i,'k'], data_pos.ix[i,'r']))   
        else: 
            ax1.scatter(data_pos.ix[i,'k'], data_pos.ix[i,'r'], color = 'b')
            ax1.annotate(data_pos.ix[i,'Name'], xy = (data_pos.ix[i,'k'], data_pos.ix[i,'r']))
        n+= 1
    ax1.set_xlabel('Absolute value of slope in up-regulated miRNAs')
    ax1.set_ylabel('Coefficient of determination R')
    ax1.set_ylim([0.6,1])
    n = 0
    for i in range(len(data_neg)):
        if threshold_neg!=None:
            if data_neg.ix[i,'k']>=threshold_neg: 
                ax2.scatter(data_neg.ix[i,'k'], data_neg.ix[i,'r'], color = 'c')
                ax2.annotate(data_neg.ix[i,'Name'], xy = (data_neg.ix[i,'k'], data_neg.ix[i,'r']))   
                n+=1
        else: 
            ax2.scatter(data_neg.ix[i,'k'], data_neg.ix[i,'r'], color = 'c')
            ax2.annotate(data_neg.ix[i,'Name'], xy = (data_neg.ix[i,'k'], data_neg.ix[i,'r']))
        n+= 1
    ax2.set_xlabel('Absolute value of slope in down-regulated miRNAs')
    ax2.set_ylabel('Coefficient of determinationR')
    ax2.set_ylim([0.6,1])
#进行时间序列分析
#包含常数和趋势项
result1 = []
for i in range(len(df_mean_f)):
    result1.append(ts.adfuller(df_mean_f.iloc[i,2:16],1,regression = 'ct')[1])
#包含常数项
result2 = []
for i in range(len(df_mean_f)):
    result2.append(ts.adfuller(df_mean_f.iloc[i,2:16],1,regression = 'c')[1]) 
#不包含
result3 = []
for i in range(len(df_mean_f)):
    result3.append(ts.adfuller(df_mean_f.iloc[i,2:16],1,regression = 'nc')[1]) 
    
result1 = np.array(result1).reshape(-1,1)
result2 = np.array(result2).reshape(-1,1)
result3 = np.array(result3).reshape(-1,1)
df_ADF = pd.DataFrame(np.concatenate([result1,result2,result3],axis=1),columns=['ct', 'c', 'nc'])

#'mmu-miR-181a-5p','mmu-miR-9-5p','mmu-miR-130a-3p','mmu-miR-181b-5p','mmu-miR-125a-5p'
def miRNA_plot(k,*names): #miRNA画折线图
    fig, ax = plt.subplots(figsize=(18, 14))
    #x = np.arange(0,14)
    x = np.array(day_array).reshape(-1,1)
    if k=='pos': data = df_mean_f_pos
    if k=='neg': data = df_mean_f_neg
    if k=='both': data = df_mean_f
    for i in names:
        temp = data[data['Name']==i]
        ax.plot(x, temp.iloc[0,2:upper], label = i)
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(day_label)
    ax.set_xlabel('Time')
    ax.set_ylabel('Expression')   
    
def demonstration1():
    fig, axes = plt.subplots(figsize=(24, 24), nrows=2, ncols=2)
    #x = np.arange(0,14)
    x = np.array(day_array).reshape(-1,1)
    for i in axes.flat[:]: #生成的axes为二维向量，无法直接套用循环，flat方法将其扁平化至一维
        i.set_xticks(x)
        i.set_xticklabels(day_label)
    if delete==0:
        ax1 = axes[0,0]
        ax1.scatter(x, df_mean_f_pos.iloc[304,2:upper], label='mmu-miR-129-1-3p', color='deepskyblue')
        regr.fit(x.reshape(-1,1),df_mean_f_pos.iloc[304,2:upper])
        ax1.plot(x, regr.predict(x.reshape(-1,1)), color='darkred')
        ax1.set_ylim([0,30])
        ax1.set_xlabel('mmu-miR-129-1-3p')
        ax1.set_ylabel('Expression')
        ax1.text(2, 25, 'A1',fontsize = 24)
        
        ax2 = axes[0,1]
        ax2.scatter(x, df_mean_f_pos.iloc[544,2:upper], label='mmu-miR-378b', color='deepskyblue')
        regr.fit(x.reshape(-1,1),df_mean_f_pos.iloc[544,2:upper])
        ax2.plot(x, regr.predict(x.reshape(-1,1)), color='darkred')
        ax2.set_ylim([0,30])
        ax2.set_xlabel('mmu-miR-378b')
        ax2.set_ylabel('Expression')
        ax2.text(2, 25, 'A2',fontsize = 24)
        
        ax3 = axes[1,0]
        ax3.scatter(x, df_mean_f_neg.iloc[129,2:upper], label = 'mmu-miR-106a-5p', color='lightblue')
        regr.fit(x.reshape(-1,1),df_mean_f_neg.iloc[129,2:upper])
        n = 2
        ax3.plot(x, regr.predict(x.reshape(-1,1)), color='darkred')
        for i in x:
            ax3.plot([i, i],[df_mean_f_neg.iloc[129,n], regr.predict(i[0])],'k--') #plt.plot([1,2,3],[1,4,5])中，第一个列表储存x的所有位置，第二个列表储存所有y的位置
            n+=1
        ax3.set_ylim([0,25])
        ax3.set_xlabel('mmu-miR-106a-5p')
        ax3.set_ylabel('Expression')
        ax3.text(2, 20, 'B1',fontsize = 24)
        
        ax4 = axes[1,1]
        ax4.scatter(x, df_mean_f_neg.iloc[97,2:upper], label = 'mmu-miR-767-5p', color='lightblue')
        regr.fit(x.reshape(-1,1),df_mean_f_neg.iloc[97,2:upper])
        n = 2
        ax4.plot(x, regr.predict(x.reshape(-1,1)), color='darkred')
        for i in x:
            ax4.plot([i, i],[df_mean_f_neg.iloc[97,n], regr.predict(i[0])],'k--')
            n+=1
        ax4.set_ylim([0,25])
        ax4.set_xlabel('mmu-miR-767-5p')
        ax4.set_ylabel('Expression')
        ax4.text(2, 20, 'B2',fontsize = 24) 
    elif delete==1:
        ax1 = axes[0,0]
        ax1.scatter(x, df_mean_f_pos.iloc[326,2:upper], label = 'mmu-miR-129-1-3p', color='deepskyblue')
        regr.fit(x.reshape(-1,1),df_mean_f_pos.iloc[326,2:upper])
        ax1.plot(x, regr.predict(x.reshape(-1,1)), color='darkred')
        ax1.set_ylim([0,30])
        ax1.set_xlabel('mmu-miR-129-1-3p')
        ax1.set_ylabel('Expression')
        ax1.text(2, 25, 'A1',fontsize = 24)
        
        ax2 = axes[0,1]
        ax2.scatter(x, df_mean_f_pos.iloc[571,2:upper], label = 'mmu-miR-378b', color='deepskyblue')
        regr.fit(x.reshape(-1,1),df_mean_f_pos.iloc[571,2:upper])
        ax2.plot(x, regr.predict(x.reshape(-1,1)), color='darkred')
        ax2.set_ylim([0,30])
        ax2.set_xlabel('mmu-miR-378b')
        ax2.set_ylabel('Expression')
        ax2.text(2, 25, 'A2',fontsize = 24)
        
        ax3 = axes[1,0]
        ax3.scatter(x, df_mean_f_neg.iloc[117,2:upper], label = 'mmu-miR-106a-5p', color='lightblue')
        regr.fit(x.reshape(-1,1),df_mean_f_neg.iloc[117,2:upper])
        n = 2
        ax3.plot(x, regr.predict(x.reshape(-1,1)), color='darkred')
        for i in x:
            ax3.plot([i, i],[df_mean_f_neg.iloc[117,n], regr.predict(i[0])],'k--') #plt.plot([1,2,3],[1,4,5])中，第一个列表储存x的所有位置，第二个列表储存所有y的位置
            n+=1
        ax3.set_ylim([0,25])
        ax3.set_xlabel('mmu-miR-106a-5p')
        ax3.set_ylabel('Expression')
        ax3.text(2, 20, 'B1',fontsize = 24)
        
        ax4 = axes[1,1]
        ax4.scatter(x, df_mean_f_neg.iloc[301,2:upper], label = 'mmu-miR-15a-5p', color='lightblue')
        regr.fit(x.reshape(-1,1),df_mean_f_neg.iloc[301,2:upper])
        n = 2
        ax4.plot(x, regr.predict(x.reshape(-1,1)), color='darkred')
        for i in x:
            ax4.plot([i, i],[df_mean_f_neg.iloc[301,n], regr.predict(i[0])],'k--')
            n+=1
        ax4.set_ylim([0,25])
        ax4.set_xlabel('mmu-miR-15a-5p')
        ax4.set_ylabel('Expression')
        ax4.text(2, 20, 'B2',fontsize = 24)

       
def demonstration9(k,para = 'k',absolute = True, zero_filter = True):
    if k == 'both': l = df_k_f.ix[:,para]
    if k == 'neg': l = df_k_neg.ix[:,para]
    if k == 'pos': l = df_k_pos.ix[:,para]
    plt.rc('font',family='SimHei',size=30)    
    l = l.reshape(1,-1)[0]
    if absolute==True:
        l = np.abs(l)
    l = np.sort(l)
    fig, ax = plt.subplots(figsize=(54, 30))
    ax.bar(range(len(l)),l,width=1, color = 'b')
    if zero_filter==True:
        l2 = l
        p = 0
        upper = np.percentile(l,80)
        lower = np.percentile(l,11)
        for i in l2:
            if i>lower and i<upper: 
                if i>=0: l2[p] = 0.01
                elif i<0: l2[p] = -0.015
            else:
                l2[p] = 0
            p+=1
        ax.bar(range(len(l2)), l2, width = 1, color ='b')    
    ax.set_xlabel('miRNAs')
    ax.set_ylabel('Absolute value of slope')
    
def demonstration10(): #绘制miRNA芯片与miRNA验证的关系
    fig, axes = plt.subplots(figsize=(50, 50), nrows=5, ncols=4)

    if delete==1:
        miRNA_list = df_valid['Name']
        n = 0
        for i in range(5):
            for j in range(0,4,2):
                currentmiRNA = miRNA_list[n]
                ax1 = axes[i,j]
                y = np.array(df[df['Name']==currentmiRNA].iloc[:,2:upper]).reshape(-1,1)
                ax1.bar(x.flat[:], y.flat[:], label=currentmiRNA, color='steelblue') #.flat[:]使[[2,3]]的格式变成[2,3]
                regr.fit(x.reshape(-1,1), y)
                ax1.plot(x, regr.predict(x.reshape(-1,1)), color='darkred', linestyle='dashed')
                ax1.set_ylim(bottom=0)
                ax1.set_xlabel(currentmiRNA)
                ax1.set_ylabel('Expression in microarray')
                ax1.set_xticks(x)
                ax1.set_xticklabels(day_label)
                ax2 = axes[i,j+1]
                y = np.array(df_valid[df_valid['Name']==currentmiRNA].iloc[:,2:upper],
                             dtype='float32').reshape(-1,1)
                ax2.bar(x.flat[:], y.flat[:], color='lightblue')
                regr.fit(x.reshape(-1,1), y)
                ax2.plot(x, regr.predict(x.reshape(-1,1)), color='darkred', linestyle='dashed')
                ax2.set_ylim(bottom=0)
                ax2.set_xlabel(currentmiRNA)
                ax2.set_ylabel('Expression in RT-qPCR')
                ax2.set_xticks(x)
                ax2.set_xticklabels(day_label)
                n += 1
                if n==len(miRNA_list):
                    break                


def demonstration12(): #画效率曲线
    l_qPCRCT_name = ['mmu-miR-130a-3p', 'mmu-miR-93-5p', 'mmu-miR-9-3p',
                     'mmu-miR-124-3p', 'mmu-miR-92a-3p', 'mmu-miR-709', 
                     'mmu-miR-181b-5p','mmu-miR-9-5p', 'mmu-miR-181a-5p', 
                     'mmu-miR-20a-5p', 'u6']
    l_qPCRCT = np.array([[16.77, 16.76, 16.80],
                          [17.72, 17.71, 17.80],
                          [18.96, 18.85, 18.81],
                          [19.90, 19.88, 19.87],
                          [20.95, 20.86, 20.94], #'miR-130a-3p'
                         [17.86, 17.80, 17.93],
                          [18.68, 18.69, 18.70],
                          [19.71, 19.65, 19.70],
                          [20.75, 20.80, 20.74],
                          [21.81, 21.85, 21.94], #'miR-93-5p'
                         [22.82, 22.94, 22.89],
                          [23.64, 23.68, 23.82],
                          [24.80, 24.63, 24.66],
                          [25.76, 25.86, 25.85],
                          [26.96, 26.84, 26.96], #'miR-9-3p'
                         [15.96, 15.75, 15.93],
                          [16.82, 16.70, 16.74],
                          [17.86, 17.76, 17.73],
                          [18.85, 18.76, 18.70],
                          [19.88, 19.78, 19.68], #'miR-124'
                         [19.94, 20.00, 19.97],
                          [20.85, 20.86, 20.81],
                          [21.96, 21.86, 21.90],
                          [22.93, 22.96, 22.96],
                          [23.98, 23.98, 23.98], #'miR-92a-3p'
                         [16.94, 16.74, 16.59],
                          [17.46, 17.53, 17.46],
                          [18.61, 18.63, 18.65],
                          [19.73, 19.68, 19.79],
                          [20.63, 20.54, 20.25],#'miR-709'
                         [14.89, 14.88, 14.85],
                          [15.77, 15.75, 15.77],
                          [16.78, 16.78, 16.79],
                          [17.85, 17.81, 17.91],
                          [18.94, 18.93, 18.89], #'miR-181b-5p'
                         [12.94, 12.76, 12.81],
                          [13.82, 13.76, 13.87],
                          [14.85, 14.80, 14.80],
                          [15.76, 15.74, 15.85],
                          [16.90, 16.88, 16.88], #'miR-9-5p' 
                         [13.54, 13.59, 13.59],
                          [14.89, 14.88, 14.85],
                          [15.77, 15.77, 15.75],
                          [16.78, 16.78, 16.79],
                          [17.85, 17.81, 17.91], #'miR-181a-5p'
                         [20.72, 20.76, 20.94],
                          [21.54, 21.61, 21.64],
                          [22.57, 22.61, 22.72],
                          [23.68, 23.68, 23.77],
                          [24.72, 24.85, 24.86], #'miR-20a-5p'
                         [21.62, 21.61, 21.59],  
                          [22.49, 22.58, 22.48],
                          [23.50, 23.63, 23.63],
                          [24.61, 24.58, 24.71],
                          [25.93, 25.84, 25.85]]) #'u6'
                             #[25.95, 25.94, 25.86],
                          #[26.85, 26.80, 26.87],
                          #[27.85, 27.81, 27.85],
                          #[28.92, 28.81, 28.78],
                          #[30.11, 29.98, 29.98], #'miR-1839-3p'
   # l_qPCRCT_average = np.array([_.sum() / 3 for _ in l_qPCRCT_original])
    fig, axes = plt.subplots(figsize=(30, 60), nrows=6, ncols=2)
    if delete==1:
        x = [1 / math.pow(2, temp + 1) for temp in range(5) for _ in range(3)]
        log2x =  [math.log(1 / math.pow(2, temp + 1), 2) for temp in range(5) for _ in range(3)]
        x_ = [1 / math.pow(2, temp + 1) for temp in range(5)]
        #xticks = [1 / math.pow(2, temp + 1) for temp in range(5)]
        xticks = [- _ - 1 for _ in range(5)]
        xtickslabel = ['1 :'+ str('%.0f' % math.pow(2, temp + 1)) for temp in range(5)]
        
        for i, j in enumerate(l_qPCRCT_name):
            m = i % 5
            if i>=5 and i!=10:
                n = 1
            elif i<5:
                n = 0
            elif i==10:
                m = 5
                n = 0 
            print([m, n])
            ax1 = axes[m, n]
            ax1.scatter(log2x, l_qPCRCT[5*i:5*(i+1)], 
                        label=j, color='black',
                        marker='+') 
            
            regr.fit(np.array(log2x).reshape(-1, 1), 
                     l_qPCRCT[5*i:5*(i+1)].flatten().reshape(-1, 1))            
            ax1.plot(log2x, regr.predict(np.array(log2x).reshape(-1, 1)), 
                     color='darkorange', linestyle='-.')

            ax1.set_ylim([12.5, 27.5])
            ax1.set_xlabel(j)
            ax1.set_ylabel('CT value')
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xtickslabel)
            r_square=regr.score(np.array(log2x).reshape(-1, 1), 
                                l_qPCRCT[5*i:5*(i+1)].flatten().reshape(-1, 1))
            k = regr.coef_[0][0]
            eff = - (math.log(- 1 / k, 2) - 1) * 100
            k = '%.4f' % k.astype(np.float32)
            
           # ax1.text(0.67, 0.95, 'Slope = ' + str(k) +'\n'
           #          + 'Efficiency = ' + str('%.1f' % eff + str('%')),
           #          transform=ax1.transAxes, va='top', fontsize=28) #可通过transform参数改变文字所在的坐标系
            ax1.text(0.67, 0.95, 'Efficiency = ' + str('%.1f' % eff + str('%')),
                     transform=ax1.transAxes, va='top', fontsize=28)
        axes[5, 1].remove()        