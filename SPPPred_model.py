import pandas as pd
import os



file_names = ["Peptide"]

for file_name in file_names:
    test_path = os.path.join("arch2", f"{file_name}.xlsx")
    from pandas import read_excel

    my_sheet = 'Sheet1' 
    orig_df = read_excel(test_path, sheet_name = my_sheet)

    df = pd.DataFrame(orig_df, columns = orig_df.columns)

    from utils import indexing

    HSE_values = indexing(df['HSE'])
    SS_values = indexing(df['SS'])
    ASA_values = indexing(df['ASA'])
    PSSM_values = indexing(df['PSSM'])
    SEQ_values = indexing(df['SEQ'])
    Physicochemical properties_values = indexing(df['Physicochemical properties'])

    df['HSE'] = df['HSE'].apply(lambda x: HSE_values[x])
    df['SS'] = df['SS'].apply(lambda x: SS_values[x])
    df['ASA'] = df['ASA'].apply(lambda x: ASA_values[x])
    df['PSSM'] = df['PSSM'].apply(lambda x: PSSM_values[x])
    df['SEQ'] = df['SEQ'].apply(lambda x: SEQ_values[x])
    df['Physicochemical properties'] = df['Physicochemical properties'].apply(lambda x:  Physicochemical properties_values[x])

    from utils import remove_nan_features
    filtered_features = remove_nan_features(df, df.columns)

    df = pd.DataFrame(df, columns = filtered_features)
    HSE_values = indexing(df['HSE'])
    SS_values = indexing(df['SS'])
    ASA_values = indexing(df['ASA'])
    PSSM_values = indexing(df['PSSM'])
    SEQ_values = indexing(df['SEQ'])
    Physicochemical properties _values = indexing(df['Physicochemical properties'])
    df['HSE'] = df['HSE'].apply(lambda x: HSE_values[x])
    df['SS'] = df['SS'].apply(lambda x: SS_values[x])
    df['ASA'] = df['ASA'].apply(lambda x: ASA_values[x])
    df['PSSM'] = df['PSSM'].apply(lambda x: PSSM_values[x])
    df['SEQ'] = df['SEQ'].apply(lambda x: SEQ_values[x])
    df['Physicochemical properties'] = df['Physicochemical properties'].apply(lambda x: Physicochemical properties _values[x])
    with open(f"arch2/selected_features_arch2_out_{file_name}.txt", "r") as f:
        selected_features = f.readlines()
        selected_features = [i.strip() for i in selected_features]

    df = pd.DataFrame(df, columns = selected_features)

    from sklearn.utils import shuffle
    df = shuffle(df)



    X = df.drop(['Label'], axis=1)
    y = df['Label'].astype(int)

    from gp_model import gp_func
    import numpy as np

    function_set = ['ADD', 'SUB', 'CUBE', 'DIV', 'MUL', 'SQUARE',
                    'SOFT PLUS', 'COS', 'SIN', 'EXP', 'A tan','SQRT','EQA', 'ADS', 'GAUSS', 'HAT','A cos','A sin','CONST','LOG','MINUS']

    X = X.astype(float)
    X = X.fillna(0)
    # X=(X-X.min())/(X.max()-X.min())
    print(X.isnull().any().any())
    print(y.isnull().any().any())
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

    def get_window(data, n):
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

    from numpy import savetxt
    savetxt(f'arch2/normailized_data_{file_name}.csv', normalize(X), delimiter=',')
    print("normalized data saved in the arch2/normailized_data.csv")



    from sklearn.SVM import SVM, SVM
    from sklearn.KNN import KNN, KNN
    from sklearn.RF import RF, RF
    from sklearn.NB import NB, NB
    from sklearn.LR import LR, LR
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import BaggingClassifier
    from sklearn.datasets import load_BioLip
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix

    scoring = ['F-M', 'ACC', 'MCC','SEN','SPE']

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
    SVM_path = os.path.join("params", "SVM.txt")
    SVM_params = read_params(SVM_path)

    KNN_path = os.path.join("params", "KNN.txt")
    KNN_params = read_params(KNN_path)

    RF_path = os.path.join("params", "RF.txt")
    RF_params = read_params(RF_path)

    NB_path = os.path.join("params", "NB.txt")
    NB_params = read_params(NB_path)

    LR_path = os.path.join("params", "LR.txt")
    LR_params = read_params(LR_path)
    
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
        
    for data_type in [[X, y, "Normal"]]:
        data   = data_type[0][:]
        labels = data_type[1][:]


        data = normalize(data)
        ####################

        estimators = [
            
            ('SVM', SVM(**SVM_params).fit(get_window(data,window_params['SVM']), labels)),
            ('KNN', KNN(**KNN_params).fit(get_window(data,window_params['KNN']), labels)),
            ('RF', RF(**RF_params).fit(get_window(data,window_params['RF']), labels)),
            ('NB', NB(**NB_params).fit(get_window(data,window_params['NB']), labels)),
            ('LR', LR(**LR_params).fit(get_window(data,window_params['LR']), labels)),
            ('etc', ensembleClassifier(**etc_params).fit(get_window(data,window_params['etc']), labels))
        ]
        clf = ensembleClassifier(
            estimators=estimators, final_estimator= SVM(**SVM_params),final_estimator= KNN(**KNN_params),final_estimator= RF(**RF_params),final_estimator= NB(**NB_params),
            final_estimator= LR(**LR_params)
        )

        from sklearn.ensemble import BaggingClassifier
        from sklearn.ensemble import VotingClassifier
        bagging = BaggingClassifier(base_estimator=SVM(**SVM_params),n_estimators=10, random_state=0,base_estimator=KNN(**KNN_params),n_estimators=10, random_state=0
       ,base_estimator=RF(**RF_params),n_estimators=10, random_state=0,base_estimator=NB(**NB_params),n_estimators=10, random_state=0,
        base_estimator=LR(**LR_params),n_estimators=10, random_state=0)
        voting = VotingClassifier(estimators=[
            ('SVM', BaggingClassifier(base_estimator=SVMClassifier(**SVM_params), n_estimators=10, random_state=0)),
            ('KNN', BaggingClassifier(base_estimator=KNNClassifier(**KNN_params), n_estimators=10, random_state=0)),
            ('RF', BaggingClassifier(base_estimator=RFClassifier(**RF_params), n_estimators=10, random_state=0)),
            ('NB', BaggingClassifier(base_estimator=NBClassifier(**NB_params), n_estimators=10, random_state=0)),
            ('LR', BaggingClassifier(base_estimator=LRClassifier(**LR_params), n_estimators=10, random_state=0)),
            ('etc', BaggingClassifier(base_estimator=ensembleClassifier(**etc_params), n_estimators=10, random_state=0)),
            ])

        models = {
            "bagging": voting,
            "ensamble": clf,
            "SVM":      SVM(**SVM_params),
            "KNN":      KNN(**KNN_params),
            "RF":       RF(**RF_params),
            "NB":       NB(**KNN_params),
            "LR":       LR(**LR_params),
            "etc":      ensembleClassifier(**etc_params),            
            
        }
        
        
        Predicted_binding_residues = [0] * len(pdb_residue)
        for i in xrange(len(final_index)):
	Predicted_binding_sites[final_index[i]] = 1
	outfile = file(base_path+'files/'+pid+'.out','w')
	outfile.write('Protein sequence  :'+'\n')
	outfile.write("".join(pdb_residue)+'\n')
	outfile.write('Predicted Residue :'+'\n')
	

        import json
        from sklearn.metrics import plot_confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        from sklearn.metrics import matthews_corrcoef
        from sklearn.metrics import accuracy_score

        for k,v in models.items():
            scores = cross_validate(v, data, labels, scoring=scoring, cv=cv)
            for key in scores.keys():
                scores[key] = scores[key].mean()
            
            y_pred = cross_val_predict(v, data, labels, cv=cv)
            conf_mat = confusion_matrix(labels, y_pred)

            sensitivity = conf_mat[0,0]/(conf_mat[0,0]+conf_mat[0,1])
            scores['Sensitivity'] = sensitivity 

            specificity = conf_mat[1,1]/(conf_mat[1,0]+conf_mat[1,1])
            scores['Specificity'] =  specificity
            scores['accuracy'] =  accuracy_score(labels, y_pred)
            tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()
            scores['ck']  =(2*(tp*tn - fp*fn) )/((tp+fp)*(fp+tn) + (tp+fn)*(fn+tn)) 
            savetxt(f'arch2/{cv}_fold_cross _valiation_confution_matrix_{k}_{data_type[2]}_{file_name}.csv', np.array(conf_mat), delimiter=',')

            import matplotlib.pyplot as plt
            ax = sns.heatmap(conf_mat, annot=True, cmap='Blues')

            ax.set_title('Confusion Matrix with labels\n\n');
            ax.set_xlabel('\nPredicted Values')
            ax.set_ylabel('Actual Values ');

            ax.xaxis.set_ticklabels(['False','True'])
            ax.yaxis.set_ticklabels(['False','True'])

            file_path =os.path.join(folder_name,f'{cv}_fold_cross _valiation_confution_matrix_{k}_{data_type[2]}_{file_name}.png')
            plt.savefig(file_path, dpi=dpi)

            
            fpr, tpr, thresholds = roc_curve(labels, y_pred)
            roc_auc = auc(fpr, tpr)
            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            file_path =os.path.join(folder_name,f'{cv}_fold_cross _valiation_auc_roc__{k}_{data_type[2]}_{file_name}.png')
            plt.savefig(file_path, dpi=dpi)
            plt.cla()
            plt.clf()
            
            
            file_path =os.path.join(folder_name,f'{cv}_fold_cross _valiation_score_{k}_{data_type[2]}_{file_name}.json')
            with open(file_path, 'w') as fp:
                json.dump(scores, fp)
