from collections import defaultdict

import pandas as pd


def load_spgemm2d_data(filename="samples.txt"):
    
    data_dict = defaultdict(lambda:[])
    
    # Parse features in file
    with open(filename, 'r') as file:
        for line in file:
            features = line.split(" ")
            for feature in features:
                
                if feature=='\n':
                    continue
                    
                feature_name, value = feature.split(":")[0], (feature.split(":")[1])
                
                # Make sure naming is consistent
                if feature_name=='layer-merge':
                    feature_name = 'merge'
                
                try:
                    value = float(value)
                except:
                    pass
                
                data_dict[feature_name].append(value)
    
    #TODO: Biases?
    
    # Make dataframe
    df = pd.DataFrame(data_dict)
    
    def get_matname_from_path(path):
        return path.split("/")[-1].strip()
    
    # Change matrix paths to names
    df['A-name'] = df['A-name'].apply(get_matname_from_path)
    df['B-name'] = df['B-name'].apply(get_matname_from_path)
    
    
    df['summed-time'] = df['bcast-A'] + df['bcast-B'] + df['merge'] + df['local-mult'] 
    
    return df



def get_problem_feature(df, matnameA, matnameB, feature):
    
    
    
    return