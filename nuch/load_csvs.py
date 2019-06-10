
# edit paths as needed
ts = './tamu_01-28_temps_smooth.csv'
meta = './tamu_01-28_metadata_clean.csv'

def load():
    '''
    loads:
        tamu_01-28_temps_smooth.csv
        tamu_01-28_metadata_clean.csv
    
    outputs:
        temps : dictionary; keys are mouse ids, values are numpy arrays.
        df : pandas DataFrame of the metadata for this set of mice.

    requires:
        csv package
        pandas package (simplicity)
        numpy package
        
    note:
        uses the file paths in the same file. If this fails,
        edit those locations; either by manually changing file, 
        or changing the variables if you import load_csvs as a module.
    '''
    import csv
    import pandas
    import numpy as np
    
    #########################
    #
    # load temperatures
    #
    
    with open(ts,'r') as f:
        csvr = csv.reader(f)
        temps_raw = list(csvr)
    #
    temps = {}
    for row in temps_raw:
        temps[row[0]] = np.array(row[1:], dtype=float)
    #
    
    #########################
    #
    # load metadata
    #
    df = pandas.read_csv(meta)
    
    return temps,df
#
