from Imports import *

def LIME(clfs,column_names,class_names,X_train,X_train_mod,X_test,protected_values,changeReferenceSet=True):
    def predict_proba_blind(X):
        X = pd.DataFrame(X, columns=column_names)
        X = X.drop(protected_values, axis=1)
        return clf.predict_proba(X.values) #clf.predict_proba(np.delete(X, [5,6,7,8,9,10], 1))
    lime_v={}
    feature_names=column_names
    np.random.seed(1)
    for k,v in clfs.items():
        clf=v
        clfType=k
        if clfType=='clfOrig' or clfType=='clfMit': 
            predict_probas=clf.predict_proba
        else:
            predict_probas=predict_proba_blind        
        if changeReferenceSet:
            if clfType=='clfOrig' or clfType=='clfOrigBlind': 
                X_train_current=X_train
            else:
                X_train_current=X_train_mod
        else:
            X_train_current=X_train
        kmeans = KMeans(init="k-means++", n_clusters=100).fit(X_train_current)
        kmeans.predict(X_train_current)
        med = kmeans.cluster_centers_
        #explainer = lime.lime_tabular.LimeTabularExplainer(X_train_current.values,class_names=class_names, feature_names = feature_names,
        explainer = lime.lime_tabular.LimeTabularExplainer(med,class_names=class_names, feature_names = feature_names,
                                                           kernel_width=3, verbose=False)
        temp = []
        for i in range(len(X_test)):
            exp = explainer.explain_instance(X_test.values[i], predict_probas, num_features=len(feature_names))
            temp.append(exp.as_map()[1])
        d={}
        for i in range(X_test.shape[1]):
            d[i]=[]
        for arr in temp: #for record
            for elem in arr: #for attribute
                d[elem[0]].append(elem[1]) #append expl. 
        for k in range(len(feature_names)):
            d[feature_names[k]] = d.pop(k)
        lime_v_1=pd.DataFrame.from_dict(d)
        lime_v[clfType]=lime_v_1
    return lime_v
    
def SHAP(clfs,column_names,X_train,X_train_mod,X_test,protected_values,changeReferenceSet=True,Tree=False):    
    def predict_proba_blind(X):
        X = pd.DataFrame(X, columns=column_names)
        X = X.drop(protected_values, axis=1)
        return clf.predict_proba(X.values) #clf.predict_proba(np.delete(X, [5,6,7,8,9,10], 1))
    shap_v={}
    for k,v in clfs.items():
        clf=v
        clfType=k
        if clfType=='clfOrig' or clfType=='clfMit': 
            predict_probas=clf.predict_proba
        else:
            predict_probas=predict_proba_blind
        if changeReferenceSet:
            if clfType=='clfOrig' or clfType=='clfOrigBlind': 
                kmeans = KMeans(init="k-means++", n_clusters=100).fit(X_train)
                kmeans.predict(X_train)
                med = kmeans.cluster_centers_
                #med = X_train.median().values.reshape((1,X_train.shape[1]))
            else:
                kmeans = KMeans(init="k-means++", n_clusters=100).fit(X_train_mod)
                kmeans.predict(X_train_mod)
                med = kmeans.cluster_centers_
                #med = X_train_mod.median().values.reshape((1,X_train_mod.shape[1]))
        else:
            kmeans = KMeans(init="k-means++", n_clusters=100).fit(X_train)
            kmeans.predict(X_train)
            med = kmeans.cluster_centers_
            #med = X_train.median().values.reshape((1,X_train.shape[1]))
        if Tree:
            explainer = shap.Explainer(clf)
        else:
            explainer = shap.KernelExplainer(predict_probas, med)#, link="logit")
        shap_values = explainer.shap_values(X_test.values)
        shap_v_1=pd.DataFrame(shap_values[1], columns=list(X_test.columns)) 
        shap_v[clfType]=shap_v_1
    return shap_v

def deltas(XAI_v,column_names,diffClfs=True,XAI_v_2=None):
    if diffClfs:
        tuples=[
        ['clfOrig','clfOrigBlind'],
        ['clfMit','clfMitBlind'],
        ['clfOrig','clfMit'],
        ['clfOrigBlind','clfMitBlind']]
    else: 
        tuples=[
        ['clfOrig','clfOrig'],
        ['clfMit','clfMit'],
        ['clfOrigBlind','clfOrigBlind'],
        ['clfMitBlind','clfMitBlind']]
    XAISum=pd.DataFrame()
    XAIMean=pd.DataFrame()
    for t in tuples:
        m=[]
        deltas = pd.DataFrame(columns=column_names)
        for col in column_names:
            if diffClfs:
                deltas[col]=np.absolute(XAI_v[t[0]][col]-XAI_v[t[1]][col])
            else:
                deltas[col]=np.absolute(XAI_v[t[0]][col]-XAI_v_2[t[1]][col])
            m.append(deltas[col].mean())
        XAIMean[t[0]+'-'+t[1]]=m
    XAIMean.set_index([column_names], inplace=True)
    return XAISum,XAIMean

def buildFinalResXAI(XAISum,XAIMean,protected_values,non_sensitive):
    res = XAIMean.T 
    res = res[protected_values]
    res = res.T
    temp=[]
    temp1=[]
    for (columnName, columnData) in res.iteritems():
        temp.append(np.absolute(res[columnName]).mean())
        temp1.append(np.std(np.absolute(res[columnName])))
    res = res.T
    res['SA-Mean'] = temp
    res['SA-Sd'] = temp1
    res = res.drop(protected_values,axis=1)
    res = res.T
    
    tempRes = XAIMean.T 
    tempRes = tempRes[non_sensitive]
    tempRes = tempRes.T
    temp=[]
    temp1=[]
    for (columnName, columnData) in tempRes.iteritems():
        temp.append(np.absolute(tempRes[columnName]).mean())
        temp1.append(np.std(np.absolute(tempRes[columnName])))
    res.loc['NSA-Mean'] = temp
    res.loc['NSA-Sd'] = temp1
    return res

def computeE(XAI_v,column_names):
    tuples=[
        ['clfOrig','clfOrigBlind'],
        ['clfMit','clfMitBlind'],
        ['clfOrig','clfMit'],
        ['clfOrigBlind','clfMitBlind']]
    XAI=pd.DataFrame()
    XAI['E'] = np.nan
    for t in tuples:
        r=[]
        deltas = pd.DataFrame(columns=column_names)
        for col in column_names:
            deltas[col]=np.absolute(XAI_v[t[0]][col]-XAI_v[t[1]][col])
        for i in range(len(deltas)):
            r.append(deltas.iloc[i].mean())
        m = np.asarray(r).mean()
        XAI.loc[t[0]+'-'+t[1]] = [round(m,3)]
    return XAI

'''
def computeE(XAI_v):
    clfs={'clfOrig':0,'clfMit':0,'clfOrigBlind':0,'clfMitBlind':0}
#    per e, aggregare ad un numero: 
#        accorpiamo somma contributi di tutti gli attributi di quel record  
#        media della sommatoria di sopra 
    for c in clfs.keys():
        s=[]
        for i in range(len(XAI_v[c])):
            s.append(XAI_v[c].iloc[i].sum())
        meanOfSum = (np.absolute(s).sum())/len(XAI_v[c])
        clfs[c]=meanOfSum
    return clfs
'''

def XAIPlots(XAI_v,column_names,diffClfs=True,XAI_v_2=None):
    markers=[
    ".",">","^","<","p","*","o","s","*","+","D","|","_","X","d","h","H",
    ".",">","^","<","p","*","o","s","*","+","D","|","_","X","d","h","H",
    ".",">","^","<","p","*","o","s","*","+","D","|","_","X","d","h","H",
    ".",">","^","<","p","*","o","s","*","+","D","|","_","X","d","h","H" 
    ]
    if diffClfs:
        tuples=[
        ['clfOrig','clfOrigBlind'],
        ['clfMit','clfMitBlind'],
        ['clfOrig','clfMit'],
        ['clfOrigBlind','clfMitBlind']]
    else: 
        tuples=[
        ['clfOrig','clfOrig'],
        ['clfMit','clfMit'],
        ['clfOrigBlind','clfOrigBlind'],
        ['clfMitBlind','clfMitBlind']]
    i=0
    for t in tuples:
        for col in column_names:
            if diffClfs:
                plt.scatter(XAI_v[t[0]][col], XAI_v[t[1]][col], s=15, alpha=0.3, marker=markers[i])
            else:
                plt.scatter(XAI_v[t[0]][col], XAI_v_2[t[1]][col], s=15, alpha=0.3, marker=markers[i])
            i+=1
            plt.xlabel(t[0])
            plt.ylabel(t[1])
        i=0
        plt.legend(column_names, bbox_to_anchor=(1,1), loc="upper left", title=t[0]+'-'+t[1])
        plt.show()