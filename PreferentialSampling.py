from Imports import *

def findFreq(target,protected,df):
    d_counts_label_0={}
    d_counts_label_1={}
    label=0
    for item in protected:
      d_counts_label_0[item]=len(df[ (df[item]==1) & (df[target]==label) ])
    label=1
    for item in protected:
      d_counts_label_1[item]=len(df[ (df[item]==1) & (df[target]==label) ])
    return d_counts_label_0,d_counts_label_1

def findMostFrequent(d,DN=None):
  highest = list(d.keys())[list(d.values()).index(max(d.values()))]
  if DN:
    if highest==DN:
      del d[highest]
      highest = list(d.keys())[list(d.values()).index(max(d.values()))]
  return highest, max(d.values())

def findLessFrequent(d):
  lowest = list(d.keys())[list(d.values()).index(min(d.values()))]
  return lowest, min(d.values())

def restList(df,preferentialSamplingSubset):
    df_all = df.merge(preferentialSamplingSubset.drop_duplicates(), how='left', indicator=True)
    restDataframe = df_all[(df_all == 'left_only').any(axis=1)]
    restDataframe=restDataframe.drop(['_merge'], 1)
    return restDataframe

#Calculate the expected size for each combination of
#v ∈ SA and c ∈ C by
# |v| x |c| / |D| 
def expectedSizes(sp,sn,fp,fn): #len(DP,DN,FP,FN,rest)
    total=sp+sn+fp+fn
    d={}
    d['DP']= round(((sp+sn)/total)*((sp+fp)))
    d['DN']= round(((sp+sn)/total)*((sn+fn)))
    d['FP']= round(((fp+fn)/total)*((sp+fp)))
    d['FN']= round(((fp+fn)/total)*((sn+fn)))
    return d 

def computeProbas(subset_X_train,clf):
  probas=clf.predict_proba(subset_X_train)
  probas_0=[]
  probas_1=[]
  for item in probas:
    probas_0.append(item[0])
    probas_1.append(item[1])
  return probas_0, probas_1

def Ranking(probas,D,Descending):
  D = D.assign(Probas=probas) #Assumiamo come classe positiva Low
  if Descending:
    sorted_df = D.sort_values(by='Probas', ascending=False)
  else:
    sorted_df = D.sort_values(by='Probas')
  return sorted_df

# UnderSampling of DN and FP = remove top 
def underSampling(D,expected):
    toRemove = len(D)-expected
    print(toRemove,'to be removed')
    return D.drop(D.index[:toRemove])     

# OverSampling of DP and FN = duplicate top, move to bottom with duplicate 
def overSampling(D,expected):
  toAdd=expected-len(D)
  print(toAdd,'to add')
  index=0
  for i in range(toAdd):
    df_new=pd.concat([D.iloc[[index]]] * 2)
    D = D.drop(D.index[index])
    frames = [D, df_new]
    D = pd.concat(frames)
    D=D.reset_index()
    D=D.drop(['index'], 1)
  return D 

def PreferentialSampling(target,protected,clf,blind,df,adClass,disClass,adAttr=None,disAttr=None):
    df.rename(columns = {'target':target}, inplace = True)
    ff=findFreq(target,protected,df)
    d_counts_label_0=ff[0]
    d_counts_label_1=ff[1]
    print('Counting labels = 0 ',d_counts_label_0)
    print('Counting labels = 1 ',d_counts_label_1)

    if adClass==0:
      if adAttr:
        DN=(disAttr,d_counts_label_1[disAttr])#disClass)
        FP=(adAttr,d_counts_label_0[adAttr])#adClass)
        DP=(disAttr,d_counts_label_0[disAttr])#adClass)
        FN=(adAttr,d_counts_label_1[adAttr])#disClass)
      else: 
        DN=findMostFrequent(d_counts_label_1)
        DP=(DN[0],d_counts_label_0[DN[0]])#findLessFrequent(d_counts_label_0) 
        FP=findMostFrequent(d_counts_label_0,DN[0])
        FN=(FP[0],d_counts_label_1[FP[0]])#findLessFrequent(d_counts_label_1)
    else:
      if adAttr:
        DN=(disAttr,d_counts_label_0[disAttr])#disClass)
        FP=(adAttr,d_counts_label_1[adAttr])#adClass)
        DP=(disAttr,d_counts_label_1[disAttr])#adClass)
        FN=(adAttr,d_counts_label_0[adAttr])#disClass)
      else:
        DN=findMostFrequent(d_counts_label_0)
        DP=(DN[0],d_counts_label_1[DN[0]]) 
        FP=findMostFrequent(d_counts_label_1,DN[0])
        FN=(FP[0],d_counts_label_0[FP[0]])
        #DN=findMostFrequent(d_counts_label_0)
        #FP=findMostFrequent(d_counts_label_1)
        #DP=findLessFrequent(d_counts_label_1) 
        #FN=findLessFrequent(d_counts_label_0)
    print('DN ',DN)
    print('FP ',FP)
    print('DP ',DP)
    print('FN ',FN)

    DN_df = df[df[DN[0]].eq(1) & df[target].eq(disClass)]
    Y_DN_df = DN_df[target].values
    DN_df = DN_df.drop(target, 1)
    DN_df=DN_df.reset_index()
    DN_df=DN_df.drop(['index'], 1)
    print('Len DN_df ',len(DN_df))
    FP_df = df[df[FP[0]].eq(1) & df[target].eq(adClass)]
    Y_FP_df = FP_df[target].values
    FP_df = FP_df.drop(target, 1)
    FP_df=FP_df.reset_index()
    FP_df=FP_df.drop(['index'], 1)
    print('Len FP_df ',len(FP_df))
    DP_df = df[df[DP[0]].eq(1) & df[target].eq(adClass)]
    Y_DP_df = DP_df[target].values
    DP_df = DP_df.drop(target, 1)
    DP_df=DP_df.reset_index()
    DP_df=DP_df.drop(['index'], 1)
    print('Len DP_df ',len(DP_df))
    FN_df = df[df[FN[0]].eq(1) & df[target].eq(disClass)]
    Y_FN_df = FN_df[target].values
    FN_df = FN_df.drop(target, 1)
    FN_df=FN_df.reset_index()
    FN_df=FN_df.drop(['index'], 1)
    print('Len FN_df ',len(FN_df))
    
    DN_df_w_Y = DN_df
    FP_df_w_Y = FP_df
    DP_df_w_Y = DP_df
    FN_df_w_Y = FN_df
    DN_df_w_Y = DN_df_w_Y.assign(target=Y_DN_df)        
    FP_df_w_Y = FP_df_w_Y.assign(target=Y_FP_df)
    DP_df_w_Y = DP_df_w_Y.assign(target=Y_DP_df)
    FN_df_w_Y = FN_df_w_Y.assign(target=Y_FN_df)
    frames = [DN_df_w_Y,FP_df_w_Y,DP_df_w_Y,FN_df_w_Y]
    preferentialSamplingSubset = pd.concat(frames)
    preferentialSamplingSubset=preferentialSamplingSubset.reset_index()
    preferentialSamplingSubset=preferentialSamplingSubset.drop(['index'], 1)
    df.rename(columns = {target:'target'}, inplace = True)
    restData = restList(df,preferentialSamplingSubset)
    print('Len restData',len(restData)) 
    expected = expectedSizes(len(DP_df),len(DN_df),len(FP_df),len(FN_df))

    if not DN_df.empty:
        if blind:
          DN_probas = computeProbas(DN_df.drop(protected, 1).values,clf)[adClass]
        else:
          DN_probas = computeProbas(DN_df.values,clf)[adClass]
        DN_df = DN_df.assign(target=Y_DN_df)        
        DN_df = Ranking(DN_probas,DN_df,True)
        DN_df=DN_df.reset_index()
        DN_df=DN_df.drop(['index'], 1)
        DN_expected = expected['DN']#expectedSize(df[df[DN[0]].eq(1)],df[df['score_text'].eq(1)],df)
        print('DN_expected ',DN_expected)
        DN_df=underSampling(DN_df,DN_expected) #DN_df=DN_df.drop(df.index[:DN_expected])     # UnderSampling of DN and FP = remove top 
        print('Len DN_df after sampling ',len(DN_df)) 
    else: 
        print('-- > DN_df empty < --')
    if not FP_df.empty:
        if blind:
          FP_probas = computeProbas(FP_df.drop(protected, 1).values,clf)[adClass]
        else:
          FP_probas = computeProbas(FP_df.values,clf)[adClass]
        FP_df = FP_df.assign(target=Y_FP_df)
        FP_df = Ranking(FP_probas,FP_df,False)
        FP_df=FP_df.reset_index()
        FP_df=FP_df.drop(['index'], 1)
        FP_expected = expected['FP']#expectedSize(df[df[FP[0]].eq(1)],df[df['score_text'].eq(0)],df)
        print('FP_expected ',FP_expected)
        FP_df=underSampling(FP_df,FP_expected) #FP_df=FP_df.drop(df.index[:FP_expected])     # UnderSampling of DN and FP = remove top 
        print('Len FP_df after sampling ',len(FP_df)) 
    else: 
        print('-- > FP_df empty < --')
    if not DP_df.empty:
        if blind:
          DP_probas = computeProbas(DP_df.drop(protected, 1).values,clf)[adClass]
        else:
          DP_probas = computeProbas(DP_df.values,clf)[adClass]
        DP_df = DP_df.assign(target=Y_DP_df)
        DP_df = Ranking(DP_probas,DP_df,False)
        DP_df=DP_df.reset_index()
        DP_df=DP_df.drop(['index'], 1)
        DP_expected = expected['DP']#expectedSize(df[df[DP[0]].eq(1)],df[df['score_text'].eq(0)],df)
        print('DP_expected ',DP_expected)
        DP_df=overSampling(DP_df,DP_expected)
        print('Len DP_df after sampling ',len(DP_df)) 
    else: 
        print('-- > DP_df empty < --')
    if not FN_df.empty:
        if blind:
          FN_probas = computeProbas(FN_df.drop(protected, 1).values,clf)[adClass]
        else:
          FN_probas = computeProbas(FN_df.values,clf)[adClass]
        FN_df = FN_df.assign(target=Y_FN_df)
        FN_df = Ranking(FN_probas,FN_df,True)
        FN_df=FN_df.reset_index()
        FN_df=FN_df.drop(['index'], 1)
        FN_expected = expected['FN']#expectedSize(df[df[FN[0]].eq(1)],df[df['score_text'].eq(1)],df)
        print('FN_expected ',FN_expected)
        FN_df=overSampling(FN_df,FN_expected)
        print('Len FN_df after sampling ',len(FN_df)) 
    else: 
        print('-- > FN_df empty < --')
    # concat * subsets 
    frames = [DN_df,FP_df,DP_df,FN_df,restData]
    df_new = pd.concat(frames)
    df_new=df_new.reset_index()
    df_new=df_new.drop(['index'], 1)
    df_new=df_new.sample(frac = 1) #Shuffle 
    return df_new
