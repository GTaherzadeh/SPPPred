#read file
import pandas as pd
import os


file_names = ["out_peptide","out_RNA", "out_DNA", "out_CBH"]

for file_name in file_names:
    test_path = os.path.join("arch2", f"{file_name}.xlsx")
    from pandas import read_excel
    # from utils import features

    my_sheet = 'Sheet1' 
    orig_df = read_excel(test_path, sheet_name = my_sheet)
    # print(orig_df.head())
    df = pd.DataFrame(orig_df, columns = orig_df.columns)

    from utils import indexing

    SS_values = indexing(df['SS'])
    AA_values = indexing(df['AA'])
    df['SS'] = df['SS'].apply(lambda x: SS_values[x])
    df['AA'] = df['AA'].apply(lambda x: AA_values[x])
    # The complete features file is available separately

    # remove Nan features 
    from utils import remove_nan_features
    filtered_features = remove_nan_features(df, df.columns)

    df = pd.DataFrame(df, columns = filtered_features)
    SS_values = indexing(df['SS'])
    AA_values = indexing(df['AA'])
    df['SS'] = df['SS'].apply(lambda x: SS_values[x])
    df['AA'] = df['AA'].apply(lambda x: AA_values[x])


    with open(f"arch2/selected_features_arch2_{file_name}.txt", "r") as f:
        selected_features = f.readlines()
        selected_features = [i.strip() for i in selected_features]

    df = pd.DataFrame(df, columns = selected_features)

    from sklearn.utils import shuffle
    df = shuffle(df)



    X = df.drop(['Label'], axis=1)
    y = df['Label'].astype(int)

    from gp_model import gp_func
    import numpy as np

    function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log',
                    'abs', 'neg', 'inv', 'max', 'min']
    # The complete function_set file is available separately

    X = X.astype(float)
    # X=(X-X.min())/(X.max()-X.min())
    gp = gp_func(X, y, function_set)
    gp_features = gp.transform(X)
    X = np.hstack((X, gp_features))


    y =y.to_numpy()

    def normal_number_for_list(num, length):
        if num >=0 and num < length:
            return num
        elif num >=length:
            return length
        else:
            return 0

    def get_window(data, n=5):
        out = np.zeros(data.shape)
        length = data.shape[0]
        half = int(n/2)
        for i in range(length):
            temp = data[normal_number_for_list(i-half, length):i].tolist() + data[i:normal_number_for_list(i+half, length)].tolist() + [data[i].tolist()]
            for t in temp:
                out[i] += t
            out[i] = out[i]/len(temp)
        
        return out

    from sklearn import preprocessing

    def normalize(data):
        scaler = preprocessing.StandardScaler().fit(data)
        return scaler.transform(data)
    import time

    from numpy import savetxt
    savetxt(f'arch2/normailized_data_{file_name}.csv', normalize(X), delimiter=',')
    print("normalized data saved in the arch1/normailized_data.csv")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVM, SVM
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import StackingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix

    scoring = ['precision_macro', 'recall_macro', 'accuracy', 'f1_micro']
    # The complete metric file is available separately



    from imblearn.under_sampling import RandomUnderSampler

    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_under, y_under = undersample.fit_resample(X, y)

    from imblearn.over_sampling import SMOTE
    SMOTE = SMOTE()
    X_SMOTE, y_SMOTE = SMOTE.fit_resample(X, y)
    from sklearn.utils import shuffle
    X_under, y_under = shuffle(X_under, y_under, random_state=0)
    X_SMOTE, y_SMOTE = shuffle(X_SMOTE, y_SMOTE, random_state=0)
    X, y             = shuffle(X, y, random_state=0)





    def read_params(path):
        params = {}
        with open(path) as f:
            lines= f.readlines()
            lines = [line.strip().split("=") for line in lines]
            for line in lines:
                try:
                    params[line[0]] = int(line[1])
                except:
                    try:
                        params[line[0]] = float(line[1])
                    except:
                        params[line[0]] = str(line[1])
        return params




    # read params
    svr_path = os.path.join("params", "svm.txt")
    svr_params = read_params(svr_path)

    knn_path = os.path.join("params", "knn.txt")
    knn_params = read_params(knn_path)

    RF_path = os.path.join("params", "RF.txt")
    RF = read_params(RF_path)

    etc_path = os.path.join("params", "etc.txt")
    etc_params = read_params(etc_path)

    window_path = os.path.join("params", "window.txt")
    window_params = read_params(window_path)

    main_path = os.path.join("params", "main.txt")
    main_params = read_params(main_path)
    cv=  main_params['cv']
    dpi = main_params['dpi']
    #####################

    folder_name = "arch2"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support
    for data_type in [[X_under, y_under, "Under"], [X_SMOTE, y_SMOTE, "SMOTE"], [X, y, "Normal"]]:
        data   = data_type[0][:]
        labels = data_type[1][:]

        X_train, X_test, y_train, y_test = train_test_split(data, labels,
                test_size=main_params['test_size'], random_state=42)


        X_train = normalize(X_train)
        X_test  = normalize(X_test)
        ####################




        estimators = [
            
            ('svm', SVM(**svm_params).fit(get_window(X_train,window_params['svm']), y_train)),
            ('RF',  RF(**RF_params).fit(get_window(X_train,window_params['RF']), y_train)),
            ('knn', KNeighborsClassifier(**knn_params).fit(get_window(X_train,window_params['knn']), y_train)),
            ('etc', ExtraTreesClassifier(**etc_params).fit(get_window(X_train,window_params['etc']), y_train))
        ]
        clf = StackingClassifier(
            estimators=estimators, final_estimator= SVM(**svm_params)
        )


        models = {
            "ensamble": clf,
            "svm":      SVM(**svm_params),
            "RF":       RF(**RF_params),
            "etc":      ExtraTreesClassifier(**etc_params),
            "knn":      KNeighborsClassifier(**knn_params) 
            
            
        }

        import json
        from sklearn.metrics import plot_confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        from sklearn.metrics import matthews_corrcoef
        from sklearn.metrics import accuracy_score

        for k,v in models.items():
            scores = {}
            start_time = time.time()
            v.fit(X_train, y_train)
            scores['fit_time'] = time.time()-start_time
            start_time = time.time()
            y_pred = v.predict(X_test)
            scores['test_time'] = time.time()-start_time
            precision, recall, fscore, _= precision_recall_fscore_support(y_test, y_pred, average='macro')
            scores['precision'] = precision
            scores['recall'] = recall
            scores['fscore'] = fscore
            conf_mat = confusion_matrix(y_test, y_pred)
            sensitivity = conf_mat[0,0]/(conf_mat[0,0]+conf_mat[0,1])
            scores['Sensitivity'] = sensitivity 

            specificity = conf_mat[1,1]/(conf_mat[1,0]+conf_mat[1,1])
            scores['Specificity'] =  specificity
            scores['accuracy'] =  accuracy_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            scores['ck']  =(2*(tp*tn - fp*fn) )/((tp+fp)*(fp+tn) + (tp+fn)*(fn+tn)) 
            savetxt(f'arch2/independent_{str(main_params["test_size"])}_confution_matrix_{k}_{data_type[2]}_{file_name}.csv', np.array(conf_mat), delimiter=',')
            import matplotlib.pyplot as plt
            ax = sns.heatmap(conf_mat, annot=True, cmap='Blues')

            ax.set_title('Confusion Matrix with labels\n\n')
            ax.set_xlabel('\nPredicted Values')
            ax.set_ylabel('Actual Values ')
            ## Ticket labels - List must be in alphabetical order
            ax.xaxis.set_ticklabels(['False','True'])
            ax.yaxis.set_ticklabels(['False','True'])
            ## Display the visualization of the Confusion Matrix.
            file_path =os.path.join(folder_name,f'independent_{str(main_params["test_size"])}_confution_matrix_{k}_{data_type[2]}_{file_name}.png')
            plt.savefig(file_path, dpi=dpi)
            
            
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)
            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            file_path =os.path.join(folder_name,f'independent_{str(main_params["test_size"])}_auc_roc__{k}_{data_type[2]}_{file_name}.png')
            plt.savefig(file_path, dpi=dpi)
            plt.cla()
            plt.clf()
            
            scores['mcc'] = matthews_corrcoef(y_test, y_pred)
            
            file_path =os.path.join(folder_name,f'independent_{str(main_params["test_size"])}_score_{k}_{data_type[2]}_{file_name}.json')
            with open(file_path, 'w') as fp:
                json.dump(scores, fp)
