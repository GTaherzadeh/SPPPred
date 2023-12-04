#read file
import pandas as pd
import os

file_names = ["out_peptide","out_RNA", "out_DNA", "out_CBH"]

for file_name in file_names:

    test_path = os.path.join(f"{file_name}.xlsx")
    from pandas import read_excel
    # from utils import features
    import time
    my_sheet = 'Sheet1' 
    orig_df = read_excel(test_path, sheet_name = my_sheet)
    # print(orig_df.head())
    df = pd.DataFrame(orig_df)

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


    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)


    import numpy as np
    import pandas as pd

    start_time = time.time()

    import numpy as np
    from s_a import simulated_annealing, objective

    selected_features = []
    for i in filtered_features:
        bounds = np.asarray([[df[i].min(), df[i].max()]])
        n_iterations = 100
        step_size = 0.1
        df['temp'] = df[i].apply(
        lambda x: simulated_annealing(objective, bounds, n_iterations, step_size, x)[0])
        try:
            if int(df['temp'].mean()[0]/ df[i].mean()) > 2 or int(df[i].mean()/df['temp'].mean()[0]):
                selected_features.append(i)
        except:
            pass

    selected_features += ['Label']
    print('selected_features', selected_features)

    # print(len(features), len(selected_features))


    textfile = open(f"arch2/selected_features_arch2_{file_name}.txt", "w")

    for element in selected_features:

        textfile.write(element + "\n")

    textfile.close()

    # print(len(features), len(selected_features))

    df = pd.DataFrame(df, columns = selected_features)


    spand_time = time.time() - start_time
    print("spand_time is: ", spand_time)

    with open(f"arch2/spand_time_select_features_{file_name}.txt" ,'w') as f:
        f.write(str(spand_time))

    df.to_csv(f'arch2/select_features_data_{file_name}.csv') 
