features = [
    'hsa_HSEu','hsa_HSEd','hsa_CN', 'SS','P(H)','P(E)', 'P(C)', 'rASA','rASA-avg',
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K','M', 'F', 'P', 'S','T',
    'W','Y','V', 'SEQ','P1', 'P2', 'P3','P4','P5','P6','P7','Label'
]


def indexing(col):
    values = {}
    count = 0
    for i in set(col):
        values[i] = count
        count += 1
    return values

def remove_nan_features(df, features):
    import numpy as np
    filtered_features = []
    for i in features:
        if not np.isnan(df[i].mean()):
            filtered_features.append(i)
    return filtered_features
