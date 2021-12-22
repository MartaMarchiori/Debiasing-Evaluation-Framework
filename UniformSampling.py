from Imports import *
from PreferentialSampling import *

# UnderSampling of DN and FP = remove top 
def underSamplingUS(D,expected):
    #D=D.sample(frac = 1) #Shuffle 
    toRemove = len(D)-expected
    print(toRemove,'to be removed')
    for i in range(toRemove):
        D=D.drop(D.sample(n=1).index)
    return D#D.drop(D.index[:toRemove])     

# OverSampling of DP and FN = duplicate top, move to bottom with duplicate 
def overSamplingUS(D,expected):
  #D=D.sample(frac = 1) #Shuffle 
  toAdd=expected-len(D)
  print(toAdd,'to add')
  #lenD=len(D)
  for i in range(toAdd):
    #index=random.randint(0, lenD)
    D = D.append(D.sample(n=1), ignore_index = True)#(D.iloc[index], ignore_index = True)
    #df_new=pd.concat([D.iloc[[index]]] * 1)
    #D = D.drop(D.index[index])
    #frames = [D, df_new]
    #D = pd.concat(frames)
    D=D.reset_index()
    D=D.drop(['index'], 1)
  return D 

def UniformSampling(target,protected,df,adClass,disClass,adAttr=None,disAttr=None):
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
        FP=findMostFrequent(d_counts_label_0)
        DP=findLessFrequent(d_counts_label_0) 
        FN=findLessFrequent(d_counts_label_1)
    else:
      if adAttr:
        DN=(disAttr,d_counts_label_0[disAttr])#disClass)
        FP=(adAttr,d_counts_label_1[adAttr])#adClass)
        DP=(disAttr,d_counts_label_1[disAttr])#adClass)
        FN=(adAttr,d_counts_label_0[adAttr])#disClass)
      else:
        DN=findMostFrequent(d_counts_label_0)
        FP=findMostFrequent(d_counts_label_1)
        DP=findLessFrequent(d_counts_label_1) 
        FN=findLessFrequent(d_counts_label_0)
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
    expected = expectedSizes(len(DP_df),len(DN_df),len(FP_df),len(FN_df),len(restData))

    if not DN_df.empty:
        DN_df = DN_df.assign(target=Y_DN_df)        
        DN_df=DN_df.reset_index()
        DN_df=DN_df.drop(['index'], 1)
        DN_expected = expected['DN']#expectedSize(df[df[DN[0]].eq(1)],df[df['score_text'].eq(1)],df)
        print('DN_expected ',DN_expected)
        DN_df=underSamplingUS(DN_df,DN_expected) #DN_df=DN_df.drop(df.index[:DN_expected])     # UnderSampling of DN and FP = remove top 
        print('Len DN_df after sampling ',len(DN_df)) 
    else: 
        print('-- > DN_df empty < --')
    if not FP_df.empty:
        FP_df = FP_df.assign(target=Y_FP_df)
        FP_df=FP_df.reset_index()
        FP_df=FP_df.drop(['index'], 1)
        FP_expected = expected['FP']#expectedSize(df[df[FP[0]].eq(1)],df[df['score_text'].eq(0)],df)
        print('FP_expected ',FP_expected)
        FP_df=underSamplingUS(FP_df,FP_expected) #FP_df=FP_df.drop(df.index[:FP_expected])     # UnderSampling of DN and FP = remove top 
        print('Len FP_df after sampling ',len(FP_df)) 
    else: 
        print('-- > FP_df empty < --')
    if not DP_df.empty:
        DP_df = DP_df.assign(target=Y_DP_df)
        DP_df=DP_df.reset_index()
        DP_df=DP_df.drop(['index'], 1)
        DP_expected = expected['DP']#expectedSize(df[df[DP[0]].eq(1)],df[df['score_text'].eq(0)],df)
        print('DP_expected ',DP_expected)
        DP_df=overSamplingUS(DP_df,DP_expected)
        print('Len DP_df after sampling ',len(DP_df)) 
    else: 
        print('-- > DP_df empty < --')
    if not FN_df.empty:
        FN_df = FN_df.assign(target=Y_FN_df)
        FN_df=FN_df.reset_index()
        FN_df=FN_df.drop(['index'], 1)
        FN_expected = expected['FN']#expectedSize(df[df[FN[0]].eq(1)],df[df['score_text'].eq(1)],df)
        print('FN_expected ',FN_expected)
        FN_df=overSamplingUS(FN_df,FN_expected)
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