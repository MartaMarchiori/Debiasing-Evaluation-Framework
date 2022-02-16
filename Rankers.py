from Imports import *
           
# https://towardsdatascience.com/machine-learning-classifiers-comparison-with-python-33149aecdbca
def rankers_evaluation(ListOfRankers,RankersNames,X,y,folds):
    # Perform cross-validation to each machine learning classifier
    CrossValidated=[]
    for i in range(len(ListOfRankers)):
        CrossValidated.append(cross_validate(ListOfRankers[i], X, y, cv=folds, scoring=scoring))
    # Create a data frame with the models perfoamnce metrics scores
    d={}
    for i in range(len(CrossValidated)):
        l=[ 
            CrossValidated[i]['test_accuracy'].mean(),
            CrossValidated[i]['test_precision'].mean(),
            CrossValidated[i]['test_recall'].mean(),
            CrossValidated[i]['test_f1_score'].mean()
        ]
        d[RankersNames[i]]=l
    models_scores_table = pd.DataFrame(d,index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    # Return models performance metrics scores data frame
    return models_scores_table

def rankers(X,X_blind,Y):
    print('Non-Blind Rankers')
    gnb = GaussianNB()
    param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
    clf = RandomizedSearchCV(gnb, param_grid, random_state=0)
    search = clf.fit(X, Y)
    gnb = search.best_estimator_
    print(gnb)

    gnbCalibrated = CalibratedClassifierCV(base_estimator=gnb) #For prediction, the base estimator, trained using all the data, is used
    gnbCalibrated.fit(X, Y)

    lr=LogisticRegression()
    param_grid = {'penalty' : ['l1', 'l2'], 'C' : np.logspace(-4, 4, 20), 'solver' : ['liblinear']}
    clf = RandomizedSearchCV(lr, param_grid, random_state=0)
    search = clf.fit(X, Y)
    lr = search.best_estimator_
    print(lr)

    lrCalibrated = CalibratedClassifierCV(base_estimator=lr) #For prediction, the base estimator, trained using all the data, is used
    lrCalibrated.fit(X, Y)

    rf = RandomForestClassifier()
    param_grid = {'n_estimators': [200, 500], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth' : [4,5,6,7,8], 'criterion' : ['gini', 'entropy'], 'bootstrap' : [True, False]}
    clf = RandomizedSearchCV(rf, param_grid, random_state=0)
    search = clf.fit(X, Y)
    rf = search.best_estimator_
    print(rf)

    rfCalibrated = CalibratedClassifierCV(base_estimator=rf) #For prediction, the base estimator, trained using all the data, is used
    rfCalibrated.fit(X, Y)

    svc = SVC()
    param_grid = {'C': [0.1,1], 'gamma': [1,0.1]}#, 'kernel': ['linear', 'rbf']}#, 'poly', 'sigmoid']}
    clf = RandomizedSearchCV(svc, param_grid, random_state=0)
    search = clf.fit(X, Y)
    search.best_params_['probability']=True
    params = search.best_params_
    svc = SVC(**params)
    print(svc)

    svcCalibrated = CalibratedClassifierCV(base_estimator=svc) #For prediction, the base estimator, trained using all the data, is used
    svcCalibrated.fit(X, Y)

    ListOfRankers=[
        gnb,
        gnbCalibrated,
        lr,
        lrCalibrated,
        rf,
        rfCalibrated,
        svc,
        svcCalibrated
    ]
    RankersNames=['Gaussian Naive Bayes','Calibrated Gaussian Naive Bayes','Logistic Regression',
                       'Calibrated Logistic Regression','Random Forest','Calibrated Random Forest',
                       'Support Vector Classifier','Calibrated Support Vector Classifier']
    print('Non-Blind Rankers Evaluation')
    m_e_not_blind = rankers_evaluation(ListOfRankers,RankersNames,X,Y,5)
    print(m_e_not_blind)

    print('Blind Rankers')
    gnbBlind = GaussianNB()
    param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
    clf = RandomizedSearchCV(gnbBlind, param_grid, random_state=0)
    search = clf.fit(X_blind, Y)
    gnbBlind = search.best_estimator_
    print(gnbBlind)

    gnbCalibratedBlind = CalibratedClassifierCV(base_estimator=gnbBlind) #For prediction, the base estimator, trained using all the data, is used
    gnbCalibratedBlind.fit(X_blind, Y)

    lrBlind=LogisticRegression()
    param_grid = {'penalty' : ['l1', 'l2'], 'C' : np.logspace(-4, 4, 20), 'solver' : ['liblinear']}
    clf = RandomizedSearchCV(lrBlind, param_grid, random_state=0)
    search = clf.fit(X_blind, Y)
    lrBlind = search.best_estimator_
    print(lrBlind)

    lrCalibratedBlind = CalibratedClassifierCV(base_estimator=lrBlind) #For prediction, the base estimator, trained using all the data, is used
    lrCalibratedBlind.fit(X_blind, Y)

    rfBlind = RandomForestClassifier()
    param_grid = {'n_estimators': [200, 500], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth' : [4,5,6,7,8], 'criterion' : ['gini', 'entropy'], 'bootstrap' : [True, False]}
    clf = RandomizedSearchCV(rfBlind, param_grid, random_state=0)
    search = clf.fit(X_blind, Y)
    rfBlind = search.best_estimator_
    print(rfBlind)

    rfCalibratedBlind = CalibratedClassifierCV(base_estimator=rfBlind) #For prediction, the base estimator, trained using all the data, is used
    rfCalibratedBlind.fit(X_blind, Y)

    svcBlind = SVC()
    param_grid = {'C': [0.1,1], 'gamma': [1,0.1]}#, 'kernel': ['linear', 'rbf']}#, 'poly', 'sigmoid']}
    clf = RandomizedSearchCV(svcBlind, param_grid, random_state=0)
    search = clf.fit(X_blind, Y)
    search.best_params_['probability']=True
    params = search.best_params_
    svcBlind = SVC(**params)
    print(svcBlind)

    svcCalibratedBlind = CalibratedClassifierCV(base_estimator=svcBlind) #For prediction, the base estimator, trained using all the data, is used
    svcCalibratedBlind.fit(X_blind, Y)

    ListOfRankersBlind=[
        gnbBlind,
        gnbCalibratedBlind,
        lrBlind,
        lrCalibratedBlind,
        rfBlind,
        rfCalibratedBlind,
        svcBlind,
        svcCalibratedBlind
    ]
    RankersNamesBlind=['Blind Gaussian Naive Bayes','Blind Calibrated Gaussian Naive Bayes','Blind Logistic Regression',
                  'Blind Calibrated Logistic Regression','Blind Random Forest','Blind Calibrated Random Forest',
                  'Blind Support Vector Classifier','Blind Calibrated Support Vector Classifier']
    print('Blind Rankers Evaluation')
    m_e_blind = rankers_evaluation(ListOfRankersBlind,RankersNamesBlind,X_blind,Y,5)
    print(m_e_blind) 

    frames = [m_e_not_blind,m_e_blind]
    m_e = pd.concat(frames, axis=1)
    m_e['Best Score'] = m_e.idxmax(axis=1)
    print('Final Rankers Evaluation')
    print(m_e)
    ListOfAllRankers=ListOfRankers+ListOfRankersBlind
    ListOfAllRankersNames=RankersNames+RankersNamesBlind
    AllRankers={}
    for i in range(len(ListOfAllRankers)):
        AllRankers[ListOfAllRankers[i]]=ListOfAllRankersNames[i]
    #AllRankers={gnb: 'Gaussian Naive Bayes', gnbCalibrated: 'Calibrated Gaussian Naive Bayes', lr: 'Logistic Regression', lrCalibrated: 'Calibrated Logistic Regression', rf: 'Random Forest', rfCalibrated: 'Calibrated Random Forest', svc: 'Support Vector Classifier', svcCalibrated: 'Calibrated Support Vector Classifier',
    #            gnbBlind: 'Blind Gaussian Naive Bayes', gnbCalibratedBlind: 'Blind Calibrated Gaussian Naive Bayes', lrBlind: 'Blind Logistic Regression', lrCalibratedBlind: 'Blind Calibrated Logistic Regression', rfBlind: 'Blind Random Forest', rfCalibratedBlind: 'Blind Calibrated Random Forest', svcBlind: 'Blind Support Vector Classifier', svcCalibratedBlind: 'Blind Calibrated Support Vector Classifier'}
    new_ke_lis = list(AllRankers.keys())
    new_val = list(AllRankers.values())
    new_pos = new_val.index(m_e['Best Score'].mode()[0]) # value from dictionary
    ranker = new_ke_lis[new_pos]
    print('------> Best Ranker: ',m_e['Best Score'].mode()[0])
    blind = False 
    if m_e['Best Score'].mode()[0] in RankersNamesBlind:
        blind = True 
    return blind, ranker 