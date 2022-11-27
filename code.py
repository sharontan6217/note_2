# -*- coding: utf-8 -*-
"""
TAN Xiao

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import math
import time
import operator
from operator import itemgetter
import matplotlib.pyplot as plt


#-----------Data Loading-----------#

input_folder = "/Users/smile/Documents/assignment 2/data/"
input_name = "data10K10"
query_name = "queries10"
input_type = ".txt"
query_type = ".txt"
output_folder = "/Users/smile/Documents/assignment 2/output/"
output_name = "output"
output_type = ".txt"
graph_folder = "/Users/smile/Documents/assignment 2/graph/"





#-----------Assignment 2 -----------

#-----------Step 1:  data loading & computation of pivots-----------

df= pd.read_csv(input_folder+input_name+input_type,  sep=" ", index_col = False, names = ["Col 1","Col 2","Col 3","Col 4","Col 5","Col 6","Col 7","Col 8","Col 9","Col 10"], na_filter=False)
df_query= pd.read_csv(input_folder+query_name+query_type,  sep=" ", index_col = False, names = ["Col 1","Col 2","Col 3","Col 4","Col 5","Col 6","Col 7","Col 8","Col 9","Col 10"], na_filter=False)


data=[]
data_total=[]
for i in range(len(df)):
    data.append(np.asarray(df.iloc[i]))
    data_total.append((np.asarray(df.iloc[i]),i))
    i+=1
#print(len(data_total))
#print(data)

query_data=[]
for i in range(len(df_query)):
    query_data.append(np.asarray(df_query.iloc[i]))
    i+=1
#print(len(query_data))
#print(query_data)

ε_original = [0.1,0.2,0.4,0.8]
k_original = [1,5,10,50,100]

def pivots(numpivots):
    global pivot
    pivot = []
    dist=[]

    for i in range (numpivots):   
        if i == 0:   
            dist_=[]
            for j in range(len(data)):
                dist_new_ = math.dist(data[j],data[i])
                dist_.append(dist_new_) 
                #print("round : ",j,i,data[i],data[j],dist_[j])   
                j+=1
            for n in range(len(dist_)):
                if dist_[n]==max(dist_):
                    print(i,n,dist_[n],max(dist_))
                    pivot.append(n)
                    dist.append([n,dist_[n]])
                else:
                    continue
                n+=1
            print("pivot[0] is: ",pivot[0],len(pivot))
                
                
        elif i == 1:
            dist_=[]
            for j in range(len(data)):        
                #print("pivot is: ",pivot[0])
                dist_new_ = math.dist(data[j],data[pivot[0]])  
                dist_.append(dist_new_)
                print("round : ",j,i,pivot[i-1],data[pivot[i-1]],data[j],dist_[j]) 
                j+=1
            for n in range(len(dist_)):
                if dist_[n]==max(dist_):
                    print(i,n,dist_[n],max(dist_))
                    pivot.append(n)
                    dist.append([n,dist_[n]])
                else:
                    continue
                n+=1

        else:
            dist_new=[]
            for j in range(len(data)):  
                print("round : ",j,i,pivot[i-1],data[pivot[i-1]],data[j],dist_[j])          
                dist_new_ = math.dist(data[j],data[pivot[i-1]])
                dist_new.append(dist_new_)
                dist_[j]=dist_[j]+dist_new[j]
                j+=1
            for n in range(len(dist_)):
                if dist_[n]==max(dist_):
                    #print(i,n,dist_[n],max(dist_))
                    pivot.append(n)
                    dist.append([n,dist_[n]])
                else:
                    continue
                n+=1
            i+=1

    with open (output_folder+output_name+output_type,"w") as f:
        f.write("--------Assigment 2------------\n")
        f.write("--------Step 1: data loading & computation of pivots------------\n")
        f.write("The pivots are:{} \n".format(pivot))
        f.write("The pivots and distances are:{} \n".format(dist))
        f.close()  

    del dist_
    del dist_new
    
    return pivot, dist

#-----------Step 2:  iDistance method-----------

def iDistance(numpivots):
    global max_dist_total, maxd, max_dist_p
    pivot_idist=[]
    pivot,dist = pivots(numpivots)
    for i in range (len(data)):
        pivot_=[]
        for p in pivot:
            dist_=math.dist(data[p],data[i])
            pivot_.append([p,dist_,i]) 
        pivot_min = sorted(pivot_,key=itemgetter(1))[:1]
        pivot_idist.append([pivot_min[0][0],pivot_min[0][1],pivot_min[0][2]])
        i+=1
    df_idist=pd.DataFrame(pivot_idist,columns=["pivot","distance","index"])

    max_dist_total=[]
    for p in pivot:
        dist_p=[]
        for j in range(len(df_idist)):
            if df_idist["pivot"][j] == p:
                dist_p.append(df_idist["distance"][j]) 
            j+=1
        max_dist_p = max(dist_p)
        print("maximum distance of each neareast p: ", p,max_dist_p)
        max_dist_total.append([p,max_dist_p])
        

    maxd=max([distance for (indicator, distance) in max_dist_total])
    print("maximum distance is: ",maxd)

    iDist=[]
    for i in range(len(data)): 
        for p in pivot:    
            dist_=math.dist(data[p],data[i])
            iDist_=i*maxd+dist_
            iDist.append([iDist_,i])
        i+=1


    with open (output_folder+output_name+output_type,"a") as f:
        f.write("--------Assigment 2------------\n")
        f.write("--------Step 2: iDistance method------------\n")
        f.write("The iDistances' Array is:{} \n".format(iDist))
        f.write("The maxd(p) is:{} \n".format(max_dist_total))
        f.write("The maxd is:{} \n".format(maxd))
        f.close()


    return iDist, max_dist_total, maxd, max_dist_p

#-----------Step 3: range similarity queries-----------

def naiveApproach(q,ε):
    averageInstanceComp_naive_=[]
    time_naive_=[]
    naiveResult=[]
    start_time = time.time()
    for i in range (len(q)):
        count = 0
        for j in range (len(data_total)):
            if math.dist(q[i],data_total[j][0])<=ε:
                naiveResult.append(data_total[j][1])
            j+=1
            count+=1
        averageInstanceComp_naive_.append(count)
        end_time = time.time()
        processTime = end_time - start_time
        time_naive_.append(processTime)
        i+=1
    averageDistance_naive = np.average(naiveResult)
    averageInstanceComp_naive = np.average(averageInstanceComp_naive_)
    time_naive = np.average(time_naive_)

    

    print(ε,averageInstanceComp_naive,time_naive)

    

    with open (output_folder+output_name+output_type,"a") as f:
        f.write("--------Assigment 2------------\n")
        f.write("--------Step 3: range similarity queries ------------\n")
        f.write("--------Step 3.1: Naive Approach------------\n")
        f.write("The Epsilon  is:{} \n".format(ε))
        f.write("The average time required to evaluate each query is:{} minutes \n".format(time_naive))
        f.write("The number of distance computations is:{} \n".format(averageInstanceComp_naive))
        f.close()

    return averageDistance_naive,averageInstanceComp_naive,time_naive



def pivotApproach(q,p,ε):
    averageInstanceComp_pivot_=[]
    time_pivot_=[]
    pivotResult=[]   
    for i in range(len(q)): 
        start_time = time.time()
        data_original=[]
        count = 0
        for t in range(len(data_total)):
            data_original.append(data_total[t])
            t+=1
        try:   
            for j in range(len(data_original)):                    
                for n in range (len(p)):
                    d = np.abs(math.dist(data[p[n]],data_original[j][0])-math.dist(data[p[n]],q[i]))
                    if d > ε:
                        del data_original[j]
                    else:
                        if math.dist(q[i],data_original[j][0]) <= ε:
                            pivotResult.append(data_original[j][1])
                        else:
                            continue
                    n+=1  
                j+=1
                count+=1               
        except IndexError:
            pass
        averageInstanceComp_pivot_.append(count)
        end_time = time.time()
        processTime = end_time - start_time
        time_pivot_.append(processTime)
        del data_original
        i+=1
        
             

    averageDistance_pivot = np.average(pivotResult)
    averageInstanceComp_pivot = np.average(averageInstanceComp_pivot_)
    time_pivot = np.average(time_pivot_)

    print(ε,averageInstanceComp_pivot,time_pivot)

    

    

    with open (output_folder+output_name+output_type,"a") as f:
        f.write("--------Assigment 2------------\n")
        f.write("--------Step 3: range similarity queries ------------\n")
        f.write("--------Step 3.2: Pivot Approach------------\n")
        f.write("The Epsilon  is:{} \n".format(ε))
        f.write("The average time required to evaluate each query is:{} minutes \n".format(time_pivot))
        f.write("The number of distance computations is:{}  \n".format(averageInstanceComp_pivot))
        f.close()

    return averageDistance_pivot,averageInstanceComp_pivot,time_pivot



def iDistanceApproach(q,p,ε):
    iDistResult=[]
    averageInstanceComp_iDist_=[]
    time_iDist_=[]
    for i in range(len(q)): 
        start_time = time.time()
        data_original=[]
        count = 0
        p_=[]
        max_d = []
        for a in range(len(p)):
            p_.append(p[a])
            a+=1
        for m in range(len(max_dist_total)):
            max_d.append(max_dist_total[m])
            m+=1
        for t in range(len(data_total)):
            data_original.append(data_total[t])
            t+=1
        try:
            for j in range(len(data_original)):
                dist_obj=math.dist(q[i],data_original[j][0])
                for n in range(len(p_)):
                    if max_d[n][0] == p_[n]:
                        d = np.abs(math.dist(q[i],data[p_[n]])-max_d[n][1])
                        #print(max_dist_total[n][1],"distance is: ",d,"ε is: ",ε)
                        if d > ε:
                            del p_[n]
                            del max_d[n]
                        else:
                                dist_=math.dist(q[i],data[p_[n]])
                                iDist_=i*maxd+dist_
                                if dist_obj >= iDist_+ε:
                                    del data_original[j]
                                elif dist_obj <= iDist_-ε:
                                    del data_original[j]                                   
                                else:
                                    iDistResult.append(p_[n])
                    else:
                        continue
                    n+=1
                    count+=1
                j+=1
        except IndexError:
            pass
        averageInstanceComp_iDist_.append(count)
        end_time = time.time()
        processTime = end_time - start_time
        time_iDist_.append(processTime)
        del p_, data_original, max_d
        i+=1
    
    #print(iDistResult)

    averageDistance_iDist = np.average(iDistResult)
    averageInstanceComp_iDist = np.average(averageInstanceComp_iDist_)
    time_iDist = np.average(time_iDist_)

    print(ε,averageInstanceComp_iDist,time_iDist)
    
    with open (output_folder+output_name+output_type,"a") as f:
        f.write("--------Assigment 2------------\n")
        f.write("--------Step 3: range similarity queries ------------\n")
        f.write("--------Step 3.3: iDistance Approach------------\n")
        f.write("The Epsilon  is:{} \n".format(ε))
        f.write("The average time required to evaluate each query is:{} minutes \n".format(time_iDist))
        f.write("The number of distance computations is:{} \n".format(averageInstanceComp_iDist))
        f.close()

    return averageDistance_iDist,averageInstanceComp_iDist,time_iDist

#-----------Step 4: kNN similarity queries-----------
def kNNnaiveApproach(q,k):
    averageInstanceComp_kNN_naive_=[]
    time_kNN_naive_=[]

    for i in range (len(q)):
        naiveDist_kNN=[]
        naiveData_kNN=[]
        naiveIndicator_kNN=[]
        start_time=time.time()
        count = 0
        for j in range (len(data_total)):
            d = math.dist(q[i],data_total[j][0])
            naiveDist_kNN.append(d)
            naiveData_kNN.append(data_total[j][0])
            naiveIndicator_kNN.append(data_total[j][1])
            j+=1
            count+=1
        df_knn = pd.DataFrame({"distance":naiveDist_kNN,"data":naiveData_kNN,"indicator":naiveIndicator_kNN}).sort_values(["distance"])
        naiveResult_kNN = df_knn[0:k]["indicator"]
        averageInstanceComp_kNN_naive_.append(count)
        end_time = time.time()
        processTime = end_time - start_time
        time_kNN_naive_.append(processTime)
        del naiveDist_kNN, naiveData_kNN, naiveIndicator_kNN
        i+=1

    averageDistance_kNN_naive = np.average(naiveResult_kNN)
    averageInstanceComp_kNN_naive = np.average(averageInstanceComp_kNN_naive_)
    time_kNN_naive = np.average(time_kNN_naive_)

    

    print(k,averageInstanceComp_kNN_naive,time_kNN_naive)

    

    with open (output_folder+output_name+output_type,"a") as f:
        f.write("--------Assigment 2------------\n")
        f.write("--------Step 4: kNN similarity queries------------\n")
        f.write("--------Step 4.1: Naive Approach------------\n")
        f.write("The k is:{} \n".format(k))
        f.write("The average time required to evaluate each query is:{} minutes  \n".format(time_kNN_naive))
        f.write("The number of distance computations is:{} \n".format(averageInstanceComp_kNN_naive))
        f.close()

    return averageDistance_kNN_naive,averageInstanceComp_kNN_naive,time_kNN_naive

def kNNpivotApproach(q,p,k):
    DistComp_kNN_pivot=[]
    time_kNN_pivot_=[]
    pivotResult_kNN=[]
    for i in range(len(q)): 
        data_original=[]
        dist_=[]
        h_dist=[]
        h_total=[]
        start_time=time.time()
        count = 0
        for t in range(len(data_total)):
            data_original.append(data_total[t])
            t+=1   
        try:     
            for j in range(len(data_original)):
                d_ = math.dist(q[i],data_original[j][0])
                dist_.append([d_,j])                               
                if len(h_dist) < k:
                    h_dist.append(d_)
                    h_total.append(dist_[j])
                else:
                    ε = max(h_dist)
                    #print(ε)
                    for n in range(len(p)):
                        d = np.abs(math.dist(data[p[n]],data_original[j][0])-math.dist(data[p[n]],q[i]))
                        if d > ε:
                            del data_original[j]
                        else:
                            if d_ <= ε:
                                h_total.append([d_,j])
                                #print("original h_total: ", h_total)
                                h_total = sorted(h_total,key=itemgetter(0))[:k]
                                #print("sorted h_total: ", h_total)
                                h = [indicator for (dist, indicator) in h_total]
                                h_dist = [dist for (dist, indicator) in h_total]
                                #print("h is: ",h)  
                                #print("distance is: ",h_dist)          
                        n+=1
                count+=1  
                j+=1     
        except IndexError:
            pass
        end_time = time.time()
        processTime = end_time - start_time
        time_kNN_pivot_.append(processTime)
        pivotResult_kNN.append(h)
        DistComp_kNN_pivot.append(count) 
        del data_original, dist_, h_total, h_dist
        i+=1

    averageDistance_kNN_pivot = np.average(pivotResult_kNN)
    averageInstanceComp_kNN_pivot = np.average(DistComp_kNN_pivot)
    time_kNN_pivot = np.average(time_kNN_pivot_)

    print(k,averageInstanceComp_kNN_pivot,time_kNN_pivot)

    

    with open (output_folder+output_name+output_type,"a") as f:
        f.write("--------Assigment 2------------\n")
        f.write("--------Step 4: kNN similarity queries------------\n")
        f.write("--------Step 4.2:  Pivot-based kNN------------\n")
        f.write("The k is:{} \n".format(k))
        f.write("The average time required to evaluate each query is:{} minutes \n".format(time_kNN_pivot))
        f.write("The number of distance computations is:{} \n".format(averageInstanceComp_kNN_pivot))
        f.close()


    return averageDistance_kNN_pivot,averageInstanceComp_kNN_pivot,time_kNN_pivot

def kNNiDistApproach(q,p,k):
    
    DistComp_kNN_iDist=[]
    time_kNN_iDist_=[]
    h=[]

    iDistResult_kNN=[]
    
    for i in range(len(q)): 
        start_time=time.time()
        count = 0 
        p_=[]
        max_d=[]
        dist_=[]
        data_original=[]
        h_dist=[]
        h_total=[]
        for m in range(len(max_dist_total)):
            max_d.append(max_dist_total[m])
            m+=1
        for t in range(len(data_total)):
            data_original.append(data_total[t])
            t+=1  
        for a in range(len(p)):
            p_.append(p[a])
            a+=1
        try:
            for j in range(len(data_original)):
                d_ = math.dist(q[i],data_original[j][0])
                dist_.append([d_,j])                               
                if len(h_dist) < k:
                    h_dist.append(d_)
                    h_total.append(dist_[j])
                    #print("initial h_total: ", h_total)
                else:
                    ε = max(h_dist)
                    #print("ε is: ",ε)
                    for pid in p_:
                        for n in range(len(max_d)):
                            if max_d[n][0] == pid:
                                d = np.abs(math.dist(q[i],data_original[pid][0])-max_d[n][1])
                                #print(max_d[n][1],"distance is: ",d,d_,"ε is: ",ε)
                                if d > ε:
                                    del p_[pid]
                                else:
                                    if d_ >= d+ε:
                                        del data_original[j]
                                    elif d_ <= d-ε:
                                        del data_original[j]
                                    else:
                                        h_total.append([d_,j])                                       
                                        #print("original h_total: ", h_total)
                                        h_total = sorted(h_total,key=itemgetter(0))[:k]
                                        #print("sorted h_total: ", h_total)
                                        h = [indicator for (dist, indicator) in h_total]
                                        h_dist = [dist for (dist, indicator) in h_total]
                                        #print("h is: ",h)  
                                        #print("distance is: ",h_dist)    
                            n+=1                          
                            count+=1
                j+=1           
        except IndexError:
            pass     
        end_time = time.time()
        processTime = end_time - start_time
        time_kNN_iDist_.append(processTime)
        iDistResult_kNN.append(h)
        DistComp_kNN_iDist.append(count)
        del p_, data_original, dist_, h_total, h_dist
        i+=1

    averageDistance_kNN_iDist = np.average(iDistResult_kNN)
    averageInstanceComp_kNN_iDist = np.average(DistComp_kNN_iDist)
    time_kNN_iDist = np.average(time_kNN_iDist_)

    print(k,averageInstanceComp_kNN_iDist,time_kNN_iDist)

    

    with open (output_folder+output_name+output_type,"a") as f:
        f.write("--------Assigment 2------------\n")
        f.write("--------Step 4: kNN similarity queries------------\n")
        f.write("--------Step 4.3: iDistance kNN Approach------------\n")
        f.write("The k is:{} \n".format(k))
        f.write("The average time required to evaluate each query is:{} minutes \n".format(time_kNN_iDist))
        f.write("The number of distance computations is:{} \n".format(averageInstanceComp_kNN_iDist))
        f.close()


    return averageDistance_kNN_iDist,averageInstanceComp_kNN_iDist,time_kNN_iDist

def diagramRange():
    naiveInstance=[]
    naiveTime=[]
    pivotInstance=[]
    pivotTime=[]
    iDistInstance=[]
    iDistTime=[]
    for ε in ε_original:
        averageDistance_naive,averageInstanceComp_naive,time_naive = naiveApproach(query_data,ε)
        naiveInstance.append(averageInstanceComp_naive)
        naiveTime.append(time_naive)
        averageDistance_pivot,averageInstanceComp_pivot,time_pivot = pivotApproach(query_data,pivot,ε)
        pivotInstance.append(averageInstanceComp_pivot)
        pivotTime.append(time_pivot )
        averageDistance_iDist,averageInstanceComp_iDist,time_iDist = iDistanceApproach(query_data,pivot,ε)
        iDistInstance.append(averageInstanceComp_iDist)
        iDistTime.append(time_iDist)

    plt.figure(1)
    plt.plot(ε_original,naiveInstance,color='blue',label='Naive Approach')
    plt.plot(ε_original,pivotInstance,color='red',label='Pivot Approach')
    plt.plot(ε_original,iDistInstance,color='pink',label='iDist Approach')
    plt.xlabel('k')
    plt.ylabel('total time for each method')
    plt.legend()
    fig=plt.gcf()
    fig.set_size_inches(15,7)
    png_name_instanceRange = 'range similarity queries_average number of distance computations per query.png'
    fig.savefig(graph_folder+png_name_instanceRange)
    plt.close()

    plt.figure(2)
    plt.plot(ε_original,naiveTime,color='blue',label='Naive Approach')
    plt.plot(ε_original,pivotTime,color='red',label='Pivot Approach')
    plt.plot(ε_original,iDistTime,color='pink',label='iDist Approach')
    plt.xlabel('k')
    plt.ylabel('total time for each method')
    plt.legend()
    fig=plt.gcf()
    fig.set_size_inches(15,7)
    png_name_timeRange = 'range similarity queries_total time for each method.png'
    plt.savefig(graph_folder+png_name_timeRange)
    plt.close()


def diagramkNN():
    kNN_naiveInstance=[]
    kNN_naiveTime=[]
    kNN_pivotInstance=[]
    kNN_pivotTime=[]
    kNN_iDistInstance=[]
    kNN_iDistTime=[]
    for k in k_original:
        averageDistance_kNN_naive,averageInstanceComp_kNN_naive,time_kNN_naive = kNNnaiveApproach(query_data,k)
        kNN_naiveInstance.append(averageInstanceComp_kNN_naive)
        kNN_naiveTime.append(time_kNN_naive)
        averageDistance_kNN_pivot,averageInstanceComp_kNN_pivot,time_kNN_pivot = kNNpivotApproach(query_data,pivot,k)
        kNN_pivotInstance.append(averageInstanceComp_kNN_pivot)
        kNN_pivotTime.append(time_kNN_pivot)
        averageDistance_kNN_iDist,averageInstanceComp_kNN_iDist,time_kNN_iDist = kNNiDistApproach(query_data,pivot,k)
        kNN_iDistInstance.append(averageInstanceComp_kNN_iDist)
        kNN_iDistTime.append(time_kNN_iDist)

    plt.figure(3)
    plt.plot(k_original,kNN_naiveInstance,color='blue',label='kNN Naive Approach')
    plt.plot(k_original,kNN_pivotInstance,color='red',label='kNN Pivot Approach')
    plt.plot(k_original,kNN_iDistInstance,color='pink',label='kNN iDist Approach')
    plt.xlabel('k')
    plt.ylabel('total time')
    plt.legend()
    fig=plt.gcf()
    fig.set_size_inches(15,7)
    png_name_instancekNN = 'kNN similarity queries_average number of distance computations per query.png'
    fig.savefig(graph_folder+png_name_instancekNN)
    plt.close()

    plt.figure(4)
    plt.plot(k_original,kNN_naiveTime,color='blue',label='kNN Naive Approach')
    plt.plot(k_original,kNN_pivotTime,color='red',label='kNN Pivot Approach')
    plt.plot(k_original,kNN_iDistTime,color='pink',label='kNN iDist Approach')
    plt.xlabel('k')
    plt.ylabel('total time')
    plt.legend()
    fig=plt.gcf()
    fig.set_size_inches(15,7)
    png_name_timekNN = 'kNN similarity queries_total time for each method.png'
    plt.savefig(graph_folder+png_name_timekNN)
    plt.close()


if __name__=='__main__':
    pivots(5)
    iDistance(10)
    diagramRange()
    diagramkNN()


                

            






        
    
    






        




