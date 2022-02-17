from Imports import *

clfsNames=['Orig','OrigBlind','Mit','MitBlind']

def print_fairness(metric_name, metric_matrix, protected_feature, bin_names):
    res=[]
    print('The *{}* group-based fairness metric for *{}* feature split '
          'are:'.format(metric_name, protected_feature))
    for grouping_i, grouping_name_i in enumerate(bin_names):
        j_offset = grouping_i + 1
        for grouping_j, grouping_name_j in enumerate(bin_names[j_offset:]):
            grouping_j += j_offset
            if metric_matrix[grouping_i, grouping_j]:
                is_not = ' NOT '  
                res.append(grouping_name_i+grouping_name_j)
                print('    * {} satisfied for "{}" and "{}" '
                      'sub-populations.'.format(is_not, grouping_name_i,
                                                grouping_name_j))
    return res

def differences(list1,list2):
    list1=clean(list1)
    list2=clean(list2)
    set_difference = set(list1).symmetric_difference(set(list2))
    list_difference = list(set_difference)
    return list_difference
    
def clean(disparateList):
    res=[]
    char_to_replace = {
                       '"': '',
                       "''": '',
                       ',': ' ',
                       '(': '',
                       ')': ''
                                }
    for item in disparateList:
        sample_string=item
        for key, value in char_to_replace.items():
            sample_string = sample_string.replace(key, value)
        res.append(sample_string)
    return res

def DataDescription(X_test,y_test,class_names,column_names,protected_values,protected_feature):
    res=pd.DataFrame(X_test, columns=column_names)
    protected_feature = protected_feature

    res[protected_feature]=res[protected_values].idxmax(axis=1)
    FATInput = np.array(res[[protected_feature]].to_records(index=False), dtype=[(protected_feature, '<U113')])

    class_names=class_names
    grouping_column = protected_feature
    grouping_indices, grouping_names = fatf_data_tools.group_by_column(
        FATInput, grouping_column, treat_as_categorical=True)

    print('The grouping based on the *{}* feature has the '
        'following distribution:'.format(grouping_column))
    for grouping_name, grouping_idx in zip(grouping_names, grouping_indices):
        print('    * "{}" grouping has {} instances.'.format(
            grouping_name, len(grouping_idx)))

    grouping_class_distribution = dict()
    for grouping_name, grouping_idx in zip(grouping_names, grouping_indices):
        sg_y = y_test[grouping_idx]
        sg_classes, sg_counts = np.unique(sg_y, return_counts=True)

        grouping_class_distribution[grouping_name] = dict()
        for sg_class, sg_count in zip(sg_classes, sg_counts):
            sg_class_name = class_names[sg_class]

            grouping_class_distribution[grouping_name][sg_class_name] = sg_count

    print('\nThe class distribution per sub-population:')
    for grouping_name, class_distribution in grouping_class_distribution.items():
        print('    * For the "{}" grouping the classes are distributed as '
            'follows:'.format(grouping_name))
        for class_name, class_count in class_distribution.items():
            print('        - The class *{}* has {} data points.'.format(
                class_name, class_count))
    return FATInput

def disparateImpactMetrics(clfs,FATInput,X_test,X_test_blind,y_test,protected_feature):
    ea={}
    eo={}
    dp={}
    i=0
    for k, v in clfs.items():
        clf=v
        if k=='clfOrig' or k=='clfMit':
            y_pred=clf.predict(X_test)
        else:
            y_pred=clf.predict(X_test_blind)
        target='target'
        confusion_matrix_per_bin, bin_names = fatf_mt.confusion_matrix_per_subgroup(FATInput, y_test, y_pred, protected_feature, treat_as_categorical=True)
        equal_accuracy_matrix = fatf_mfm.equal_accuracy(confusion_matrix_per_bin)
        eaTemp = print_fairness('Equal Accuracy', equal_accuracy_matrix, protected_feature, bin_names)
        equal_opportunity_matrix = fatf_mfm.equal_opportunity(confusion_matrix_per_bin)
        eoTemp = print_fairness('Equal Opportunity', equal_opportunity_matrix, protected_feature, bin_names)
        demographic_parity_matrix = fatf_mfm.demographic_parity(confusion_matrix_per_bin)
        dpTemp = print_fairness('Demographic Parity', demographic_parity_matrix, protected_feature, bin_names)
        ea[clfsNames[i]]=eaTemp
        eo[clfsNames[i]]=eoTemp
        dp[clfsNames[i]]=dpTemp
        i+=1
    resEA = computeDeltas(ea)
    resEO = computeDeltas(eo)
    resDP = computeDeltas(dp)
    resDisparateImp = {'Equal Accuracy':resEA, 'Equal Opportunity':resEO, 'Demographic Parity':resDP}
    return resDisparateImp

def computeDeltas(dm):
    res={}
    res['Delta O-OB'] = differences(clean(dm['Orig']),clean(dm['OrigBlind']))
    res['Delta M-MB'] = differences(clean(dm['Mit']),clean(dm['MitBlind']))
    res['Delta O-M'] = differences(clean(dm['Orig']),clean(dm['Mit']))
    res['Delta OB-MB'] = differences(clean(dm['OrigBlind']),clean(dm['MitBlind']))
    return res

def standardMetricsPerGroup(clfs,column_names,target,X_test,y_test,protected_feature,protected_values):
    def predict_blind(X):
        return clf.predict(X.drop(protected_values, axis=1))
    
    res=pd.DataFrame(X_test, columns=column_names)
    res = res.assign(target=y_test)
    res[protected_feature]=res[protected_values].idxmax(axis=1)
    gb = res.groupby(protected_feature)    
    by_protected_attr = [gb.get_group(x) for x in gb.groups]
    metric_scores=[]
    target='target'
    for k, v in clfs.items():
        precisions={}
        recalls={}
        clf=v
        for i in range(len(by_protected_attr)):
            y_test_r = by_protected_attr[i][target].values 
            X_test_r = by_protected_attr[i].drop([target,protected_feature], 1)
            if k=='clfOrig' or k=='clfMit':
                y_pred=clf.predict(X_test_r)
            else:
                y_pred=predict_blind(X_test_r)
            precisions[protected_values[i]]=precision_score(y_pred, y_test_r, average='macro')
            recalls[protected_values[i]]=recall_score(y_pred, y_test_r, average='macro')
        metric_score={
            'Precision':precisions,
            'Recall':recalls,
        }
        metric_scores.append(metric_score) 

    for m in ['Precision','Recall']:
        labels = protected_values

        x = np.arange(len(protected_values))  
        
        fig, ax = plt.subplots()
        rects1 = ax.plot(list(metric_scores[0][m].values()), linestyle='solid', label='Orig')
        rects2 = ax.plot(list(metric_scores[1][m].values()), linestyle='dotted', label='OrigBlind')
        rects3 = ax.plot(list(metric_scores[2][m].values()), linestyle='dashed', label='Mit')
        rects4 = ax.plot(list(metric_scores[3][m].values()), linestyle='dashdot', label='MitBlind')

        ax.set_ylabel(m)
        ax.set_title(m)
        ax.set_xticks(x)
        ax.set_xticklabels(labels,rotation=40,ha='right')
        ax.legend()

        plt.show()