import numpy as np
#no. of columns
m=17
#no. of rows
n=18
dp = np.zeros([18,17],dtype=int)
print(dp)
#seeing final size
print(len(dp))  #rows
print(len(dp[0]))  #columns
#getting random characters
list=['a','c','g','t']

import random
seq1=[]
seq2=[]
for i in range(0,16):
    n=random.randint(0,3)
    for i in seq1:
        if seq1.count(i)>4:
                n=random.randint(0,3)          
    seq1.append(list[n])
print(seq1)

        

for i in range(0,16):
    n=random.randint(0,3)
    for i in seq1:
        if seq1.count(i)>4:
                n=random.randint(0,3) 
    seq2.append(list[n])
print(seq2)

for i in range(0,m-1):
    #match add 5
    if(seq1[i]==seq2[i]):
        print("matches")
        print(seq1[i])
        print(seq2[i])
        print(i)
        dp[i+1][i+1]= dp[i][i]+5
    else:
        #mismatchh  subtract 4 and find max of top and left cells
        val1=dp[i-1,i]-4
        val2=dp[i,i-1]-4
        if(val1>val2):
            new=val1
        else:
            new=val2
        dp[i+1][i+1]=new
print(dp)