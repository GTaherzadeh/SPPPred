import os 
import numpy as np

file_names = ["out_peptide","out_RNA", "out_DNA", "out_CBH"]
for file_name in file_names:
    file_path = os.path.join('.', f'{file_name}.txt')


    data = []

    with open(file_path) as f:
        lines = f.readlines()
        lines = [line.strip().split('\t') for line in lines]
        not_wanted_cols_namse = ['# name', 'no', 'A', 'B', 'C', 'D' ]
        not_wanted_cols_namse = [ i for i in not_wanted_cols_namse if i in lines[0]]
        not_wanted_cols_indices = [ lines[0].index(i) for i in not_wanted_cols_namse]
        for t in lines[:]:
            temp = []
            for i in range(len(lines[0])):
                if i not in  not_wanted_cols_indices:
                    temp.append(t[i])
            data.append(temp)
        
    # import pandas as pd
    import pandas as pd

    
    # with indices and columns specified
    df = pd.DataFrame(data[1:],  columns =data[0])

    df.to_excel(f'{file_name}.xlsx', index=False)
