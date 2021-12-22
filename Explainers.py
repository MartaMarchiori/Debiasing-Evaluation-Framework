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
            if clfType=='clfOrig' or clfType=='clfMit': 
                X_train_current=X_train
            else:
                X_train_current=X_train_mod
        else:
            X_train_current=X_train
        explainer = lime.lime_tabular.LimeTabularExplainer(X_train_current.values,class_names=class_names, feature_names = feature_names,
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
            if clfType=='clfOrig' or clfType=='clfMit': 
                med = X_train.median().values.reshape((1,X_train.shape[1]))
            else:
                med = X_train_mod.median().values.reshape((1,X_train_mod.shape[1]))
        else:
            med = X_train.median().values.reshape((1,X_train.shape[1]))
        if Tree:
            explainer = shap.Explainer(clf)
        else:
            explainer = shap.KernelExplainer(predict_probas, med)#, link="logit")
        shap_values = explainer.shap_values(X_test.values)
        shap_v_1=pd.DataFrame(shap_values[1], columns=list(X_test.columns)) 
        shap_v[clfType]=shap_v_1
    return shap_v

def deltas(shap_v,column_names):
    tuples=[
    ['clfOrig','clfOrigBlind'],
    ['clfMit','clfMitBlind'],
    ['clfOrig','clfMit'],
    ['clfOrigBlind','clfMitBlind']]
    XAISum=pd.DataFrame()
    XAIMean=pd.DataFrame()
    for t in tuples:
        m=[]
        s=[]
        deltas = pd.DataFrame(columns=column_names)
        sums = pd.DataFrame(columns=column_names)
        for col in column_names:
            deltas[col]=np.absolute(shap_v[t[0]][col]-shap_v[t[1]][col])
            m.append(deltas[col].mean())
            sums[col]=shap_v[t[0]][col]+shap_v[t[1]][col]
            s.append(sums[col].sum())
        XAIMean[t[0]+'-'+t[1]]=m
        XAISum[t[0]+'-'+t[1]]=s
    XAISum.set_index([column_names], inplace=True)
    XAIMean.set_index([column_names], inplace=True)
    return XAISum,XAIMean

def buildFinalResXAI(XAISum,XAIMean,protected_values,non_sensitive):
    res = XAISum.T 
    res = res[protected_values]
    res = res.T
    temp=[]
    for (columnName, columnData) in res.iteritems():
        temp.append(np.absolute(res[columnName]).mean())
    res = res.T
    res['SA-Sum'] = temp
    res = res.drop(protected_values,axis=1)
    res = res.T
    tempRes = XAIMean.T 
    tempRes = tempRes[protected_values]
    tempRes = tempRes.T
    temp=[]
    for (columnName, columnData) in tempRes.iteritems():
        temp.append(np.absolute(tempRes[columnName]).mean())
    res.loc['SA-Mean'] = temp

    tempRes = XAISum.T 
    tempRes = tempRes[non_sensitive]
    tempRes = tempRes.T
    temp=[]
    for (columnName, columnData) in tempRes.iteritems():
        temp.append(np.absolute(tempRes[columnName]).mean())
    res.loc['NSA-Sum'] = temp
    tempRes = XAIMean.T 
    tempRes = tempRes[non_sensitive]
    tempRes = tempRes.T
    temp=[]
    for (columnName, columnData) in tempRes.iteritems():
        temp.append(np.absolute(tempRes[columnName]).mean())
    res.loc['NSA-Mean'] = temp
    
    #resSHAP['Delta O-OB'] = pd.Series.abs(resSHAP['clfOrig'] - resSHAP['clfOrigBlind'])
    #resSHAP['Delta M-MB'] = pd.Series.abs(resSHAP['clfMit'] - resSHAP['clfMitBlind'])
    #resSHAP['Delta O-M'] = pd.Series.abs(resSHAP['clfOrig'] - resSHAP['clfMit'])
    #resSHAP['Delta OB-MB'] = pd.Series.abs(resSHAP['clfOrigBlind'] - resSHAP['clfMitBlind'])
    return res

def XAIPlots(XAI_v,column_names,diffClfs=True):
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
    for t in tuples:
        for col in column_names:
            plt.scatter(XAI_v[t[0]][col], XAI_v[t[1]][col])
        plt.legend(column_names, bbox_to_anchor=(1,1), loc="upper left", title=t[0]+'-'+t[1])
        plt.show()