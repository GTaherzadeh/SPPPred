import pandas as pd
import os

test_path = os.path.join("data", "Test.Set.xlsx")
from pandas import read_excel
from utils import features
import time
my_sheet = 'Sheet1' 
orig_df = read_excel(test_path, sheet_name = my_sheet)

df = pd.DataFrame(orig_df, columns = features)

from utils import indexing

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
df['Physicochemical properties'] = df['Physicochemical properties'].apply(lambda x:  Physicochemical properties_values[x])

from utils import remove_nan_features
filtered_features = remove_nan_features(df, features)

df = pd.DataFrame(df, columns = filtered_features)
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
df['Physicochemical properties'] = df['Physicochemical properties'].apply(lambda x:Physicochemical properties_values[x])

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import numpy as np
import pandas as pd

start_time = time.time()

import numpy as np
from numpy import array

selected_features += ['Label']
print('selected_features', selected_features)

print(len(features), len(selected_features))


textfile = open("selected_features_arch2.txt", "w")

for element in selected_features:

    textfile.write(element + "\n")

textfile.close()

print(len(features), len(selected_features))

df = pd.DataFrame(df, columns = selected_features)


spand_time = time.time() - start_time
print("spand_time is: ", spand_time)

with open("arch2/spand_time_select_features.txt" ,'w') as f:
    f.write(str(spand_time))

df.to_csv('arch2/select_features_data.csv')



