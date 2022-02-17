from Imports import *
from PreferentialSampling import *

clfs_=['CLF Original','CLF Original Blind','CLF Mitigated','CLF Mitigated Blind']

def model_evaluation_orig_vs_mit(clfOriginal, clfOrigBlind, clfMitigated, clfMitigatedBlind, X_test, X_test_blind, y_test, folds):       
    models=[clfOriginal, clfOrigBlind, clfMitigated, clfMitigatedBlind]
    scores=[]
    y_pred=clfOriginal.predict(X_test)
    scores.append(f1_score(y_test, y_pred))
    y_pred=clfOrigBlind.predict(X_test_blind)
    scores.append(f1_score(y_test, y_pred))
    y_pred=clfMitigated.predict(X_test)
    scores.append(f1_score(y_test, y_pred))
    y_pred=clfMitigatedBlind.predict(X_test_blind)
    scores.append(f1_score(y_test, y_pred))
    d={}
    for i in range(len(models)):
        d[clfs_[i]]=scores[i]
    models_scores_table = pd.DataFrame(d,index=['F1 Score'])
    return models_scores_table

def inputDiscrimination(df,target,protected_values,adClass,disClass,adAttr=None,disAttr=None):
    ff=findFreq(target,protected_values,df)
    d_counts_label_0=ff[0]
    d_counts_label_1=ff[1]

    if adClass==0:
      if adAttr:
        DN=(disAttr,d_counts_label_1[disAttr])
        FP=(adAttr,d_counts_label_0[adAttr])
        DP=(disAttr,d_counts_label_0[disAttr])
        FN=(adAttr,d_counts_label_1[adAttr])
      else: 
        DN=findMostFrequent(d_counts_label_1)
        DP=(DN[0],d_counts_label_0[DN[0]]) 
        FP=findMostFrequent(d_counts_label_0,DN[0])
        FN=(FP[0],d_counts_label_1[FP[0]])
    else:
      if adAttr:
        DN=(disAttr,d_counts_label_0[disAttr])
        FP=(adAttr,d_counts_label_1[adAttr])
        DP=(disAttr,d_counts_label_1[disAttr])
        FN=(adAttr,d_counts_label_0[adAttr])
      else:
        DN=findMostFrequent(d_counts_label_0)
        DP=(DN[0],d_counts_label_1[DN[0]]) 
        FP=findMostFrequent(d_counts_label_1,DN[0])
        FN=(FP[0],d_counts_label_0[FP[0]])
    return FP[0],DN[0]

def Discrimination(df,target,positiveClass,fav,disf):
    #count of + in favoured over total favoured - count of + in disf over total disf
    count_pos_fav = len(df[df[fav].eq(1) & df[target].eq(positiveClass)])
    count_fav = len(df[df[fav].eq(1)])
    count_pos_disf = len(df[df[disf].eq(1) & df[target].eq(positiveClass)])
    count_disf = len(df[df[disf].eq(1)])
    print('Favoured attribute value: [',fav,'] Freq.: [',count_fav,'] Labeled as + class: [',count_pos_fav,']')
    print('Unfavoured attribute value: [',disf,'] Freq.: [',count_disf,'] Labeled as + class: [',count_pos_disf,']')
    return (count_pos_fav/count_fav) - (count_pos_disf/count_disf)

def Classifiers(clfInput,param_grid,target,column_names,X_train,X_train_mod,X_train_blind,X_train_mod_blind,y_train,y_train_mod,X_test, X_test_blind, y_test, protected_values, adClass,disClass, adAttr=None,disAttr=None):
    print('Original Train Dimension =',len(X_train))
    print('Mitigated Train Dimension =',len(X_train_mod))
    print('Test Dimension =',len(X_test))

    clf = RandomizedSearchCV(clfInput, param_grid, random_state=0)
    search = clf.fit(X_train, y_train)
    clfOrig = search.best_estimator_

    clf = RandomizedSearchCV(clfInput, param_grid, random_state=0)
    search = clf.fit(X_train_mod, y_train_mod)
    clfMit = search.best_estimator_

    clf = RandomizedSearchCV(clfInput, param_grid, random_state=0)
    search = clf.fit(X_train_blind, y_train)
    clfOrigBlind = search.best_estimator_

    clf = RandomizedSearchCV(clfInput, param_grid, random_state=0)
    search = clf.fit(X_train_mod_blind, y_train_mod)
    clfMitBlind = search.best_estimator_

    res=model_evaluation_orig_vs_mit(clfOrig, clfOrigBlind, clfMit, clfMitBlind, X_test, X_test_blind, y_test, 5)
        
    discriminations=[]
    print('---> CLF Original: ')
    clf=clfOrig
    y_pred=clf.predict(X_test)
    X_test_input_disc = pd.DataFrame(X_test, columns=column_names)
    X_test_input_disc[target] = y_pred
    if adAttr:
        fav=adAttr
        disf=disAttr
    else: 
        fav,disf=inputDiscrimination(X_test_input_disc,target,protected_values,adClass,disClass)
    discOrig=Discrimination(X_test_input_disc,target,adClass,fav,disf)

    print('---> CLF Mitigated: ')
    clf=clfMit 
    y_pred=clf.predict(X_test)
    X_test_input_disc = pd.DataFrame(X_test, columns=column_names)
    X_test_input_disc[target] = y_pred
    if adAttr:
        fav=adAttr
        disf=disAttr
    else: 
        fav,disf=inputDiscrimination(X_test_input_disc,target,protected_values,adClass,disClass)
    discMit=Discrimination(X_test_input_disc,target,adClass,fav,disf)

    print('---> CLF Original Blind: ')
    clf=clfOrigBlind 
    y_pred=clf.predict(X_test_blind)
    X_test_input_disc = pd.DataFrame(X_test, columns=column_names)
    X_test_input_disc[target] = y_pred
    if adAttr:
        fav=adAttr
        disf=disAttr
    else: 
        fav,disf=inputDiscrimination(X_test_input_disc,target,protected_values,adClass,disClass)
    discOrigBlind=Discrimination(X_test_input_disc,target,adClass,fav,disf)

    print('---> CLF Mitigated Blind: ')
    clf = clfMitBlind
    y_pred=clf.predict(X_test_blind)
    X_test_input_disc = pd.DataFrame(X_test, columns=column_names)
    X_test_input_disc[target] = y_pred
    if adAttr:
        fav=adAttr
        disf=disAttr
    else: 
        fav,disf=inputDiscrimination(X_test_input_disc,target,protected_values,adClass,disClass)
    discMitBlind=Discrimination(X_test_input_disc,target,adClass,fav,disf)

    discriminations = [discOrig,discOrigBlind,discMit,discMitBlind]
    res.loc['Discrimination'] = discriminations
    res['Best Score'] = res.idxmax(axis=1)

    clfs=['CLF Original','CLF Original Blind','CLF Mitigated','CLF Mitigated Blind']
    discs=res.loc['Discrimination'][clfs].values 
    idxmin = np.where(discs == np.amin(discs))[0][0]
    res.at['Discrimination','Best Score']=clfs[idxmin]

    res['Delta O-OB'] = pd.Series.abs(res['CLF Original'] - res['CLF Original Blind'])
    res['Delta M-MB'] = pd.Series.abs(res['CLF Mitigated'] - res['CLF Mitigated Blind'])
    res['Delta O-M'] = pd.Series.abs(res['CLF Original'] - res['CLF Mitigated'])
    res['Delta OB-MB'] = pd.Series.abs(res['CLF Original Blind'] - res['CLF Mitigated Blind'])

    return clfOrig,clfOrigBlind,clfMit,clfMitBlind,res 

def plotDisc(res, path_res, name):
    markers=[".",">","p","*"]
    i=0
    for x, y in zip(res.loc['F1 Score'][clfs_], res.loc['Discrimination'][clfs_]):
        plt.scatter(x, y, marker=markers[i])
        i+=1
    plt.xlabel('F1 Score')
    plt.ylabel('Discrimination')
    plt.legend(clfs_)
    plt.savefig(path_res+name+"F1Score.jpg")
    plt.show()
