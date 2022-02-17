from Imports import *

def recognize_features_type(df, class_name):    
    integer_features = list(df.select_dtypes(include=['int64']).columns)
    double_features = list(df.select_dtypes(include=['float64']).columns)
    string_features = list(df.select_dtypes(include=['object']).columns)
    type_features = {
        'integer': integer_features,
        'double': double_features,
        'string': string_features,
    }
    features_type = dict()
    for col in integer_features:
        features_type[col] = 'integer'
    for col in double_features:
        features_type[col] = 'double'
    for col in string_features:
        features_type[col] = 'string'
        
    return type_features, features_type


def set_discrete_continuous(features, type_features, class_name, discrete=None, continuous=None):
    
    if discrete is None and continuous is None:
        discrete = type_features['string']
        continuous = type_features['integer'] + type_features['double']
        
    if discrete is None and continuous is not None:
        discrete = [f for f in features if f not in continuous]
        continuous = list(set(continuous + type_features['integer'] + type_features['double']))
        
    if continuous is None and discrete is not None:
        continuous = [f for f in features if f not in discrete and (f in type_features['integer'] or f in type_features['double'])]
        discrete = list(set(discrete + type_features['string']))
    
    discrete = [f for f in discrete if f != class_name] # + [class_name]
    continuous = [f for f in continuous if f != class_name]
    return discrete, continuous


def label_encode_sensitive(df, columns):
    for col in columns:
        df1=pd.get_dummies(df[col],prefix=col)
        df=pd.concat([df, df1], axis=1)
    return df


def label_encode(df, columns, label_encoder=None):
    df_le = df.copy(deep=True)
    new_le = label_encoder is None
    label_encoder = dict() if new_le else label_encoder
    for col in columns:
        if new_le:
            le = LabelEncoder()
            df_le[col] = le.fit_transform(df_le[col])
            label_encoder[col] = le
        else:
            le = label_encoder[col]
            df_le[col] = le.transform(df_le[col])
    return df_le, label_encoder


def prepare_german_dataset(filename, path_data, sensitive):

    df = pd.read_csv(path_data + filename, delimiter=',')

    columns = df.columns
    df.rename(columns = {'default':'target'}, inplace = True)

    type_features, features_type = recognize_features_type(df, 'target')

    discrete = ['installment_as_income_perc', 'present_res_since', 'credits_this_bank', 'people_under_maintenance']
    discrete, continuous = set_discrete_continuous(columns, type_features, 'target', discrete, continuous=None)
    
    non_sensitive = [elem for elem in discrete if elem not in sensitive]

    df = label_encode_sensitive(df, sensitive)
    df, label_encoder = label_encode(df, non_sensitive)
    
    df.drop(sensitive, axis=1, inplace=True)

    discrete = non_sensitive
    discrete.append('foreign_worker_no')
    discrete.append('foreign_worker_yes')
    discrete.append('target')
    df_Discrete = df.loc[:, df.columns.isin(discrete)]
    df_ToScale = df.loc[:, ~df.columns.isin(discrete)]
    scaler = StandardScaler()

    df_ToScale = scaler.fit_transform(df_ToScale)
    df_ToScale = pd.DataFrame(df_ToScale)
    df_ToScale.columns = range(0, df_ToScale.columns.size)
    df_ToScale.columns = ['duration_in_month','credit_amount','age']
    df = pd.concat([df_ToScale.reset_index(drop=True), df_Discrete.reset_index(drop=True)], axis= 1)
    df

    return non_sensitive,df


def prepare_adult_dataset(filename, path_data, sensitive):

    df = pd.read_csv(path_data + filename, delimiter=',', skipinitialspace=True)

    del df['fnlwgt']
    del df['education-num']

    for col in df.columns:
        if '?' in df[col].unique():
            df[col][df[col] == '?'] = df[col].value_counts().index[0]

    columns = df.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    df = df[columns]
    df.rename(columns = {'class':'target'}, inplace = True)

    dictionary={'<=50K': 0, '>50K': 1}
    df['target']=df['target'].map(dictionary)

    type_features, features_type = recognize_features_type(df, 'target')

    discrete, continuous = set_discrete_continuous(columns, type_features, 'target', discrete=None, continuous=None)

    non_sensitive = [elem for elem in discrete if elem not in sensitive]

    df = label_encode_sensitive(df, sensitive)
    df, label_encoder = label_encode(df, non_sensitive)
    
    df.drop(sensitive, axis=1, inplace=True)

    return df


def prepare_compass_dataset(filename, path_data, sensitive):

    df = pd.read_csv(path_data + filename, delimiter=',', skipinitialspace=True)

    df=df.loc[df['days_b_screening_arrest'] <= 30]
    df=df.loc[df['days_b_screening_arrest'] >= -30]
    df=df.loc[df['is_recid'] != -1]
    df=df.loc[df['c_charge_degree'] != "O"]   
    df=df.loc[df['score_text'] != 'N/A']

    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])
    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])
    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)
    df['length_of_stay'] = df['length_of_stay'].astype(int)
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)

    def get_class(x):
        if x < 7:
            return 'Medium-Low'
        else:
            return 'High'
    df['class'] = df['decile_score'].apply(get_class)
    
    columns = [ 'age', 'sex', 'race',  'priors_count', 'days_b_screening_arrest', 
                'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 
                'length_of_stay','class' ]

    df = df[columns]

    df.rename(columns = {'class':'target'}, inplace = True)
    dictionary={'Medium-Low': 0, 'High': 1}
    df['target']=df['target'].map(dictionary)
    
    df.loc[(df.race == 'Native American'), 'race'] = 'Native-American'

    type_features, features_type = recognize_features_type(df, 'target')

    discrete, continuous = set_discrete_continuous(columns, type_features, 'target', discrete=None, continuous=None)

    non_sensitive = [elem for elem in discrete if elem not in sensitive]

    df = label_encode_sensitive(df, sensitive)
    df, label_encoder = label_encode(df, non_sensitive)
    
    df.drop(sensitive, axis=1, inplace=True)

    return df


def prepare_toy_dataset(df, sensitive):

    columns = df.columns
    df.rename(columns = {'Cl':'target'}, inplace = True)

    type_features, features_type = recognize_features_type(df, 'target')

    discrete = []
    discrete, continuous = set_discrete_continuous(columns, type_features, 'target', discrete, continuous=None)
    
    non_sensitive = [elem for elem in discrete if elem not in sensitive]

    df = label_encode_sensitive(df, sensitive)
    df, label_encoder = label_encode(df, non_sensitive)
    
    df.drop(sensitive, axis=1, inplace=True)

    return df


def prepare_for_sampling(df,protected_values):
    target='target'
    classes = df[target].unique()
    column_names = df.columns.values.tolist()
    column_names.remove(target)
    column_names_blind = [ele for ele in column_names if ele not in protected_values]

    Y = df[target].values 
    X = df.drop(target, 1)
    X_blind = X
    X_blind = X_blind.drop(protected_values, axis=1) 

    X_train, X_test, y_train, y_test = train_test_split(X.values, Y, test_size=0.3, random_state=0)

    df = pd.DataFrame(X_train, columns=column_names)
    df = df.assign(target=y_train)

    return X,Y,X_blind,X_train,X_test,y_train,y_test,df 


def prepare_for_classification(df_new,X_train,X_test,column_names,protected_values,PS=False):
    df_new.rename(columns = {'target':'target'}, inplace = True) 

    X_train = pd.DataFrame(X_train, columns=column_names)
    column_names_blind = [ele for ele in column_names if ele not in protected_values]
    X_train_blind = X_train 
    X_train_blind = X_train_blind.drop(protected_values, axis=1) 
    if PS:
        toDrop=['target','Probas']
    else:
        toDrop=['target']
    y_train_mod = df_new['target'].values 
    X_train_mod = df_new.drop(toDrop, 1)

    X_train_mod_blind = df_new 
    X_train_mod_blind = X_train_mod_blind.drop(protected_values, axis=1)
    X_train_mod_blind = X_train_mod_blind.drop(toDrop, 1)

    X_test = pd.DataFrame(X_test, columns=column_names)

    X_test_blind = X_test.drop(protected_values, axis=1)
    X_test_blind = pd.DataFrame(X_test_blind, columns=column_names_blind)

    return X_train,X_train_blind,X_train_mod,X_train_mod_blind,y_train_mod,X_test,X_test_blind