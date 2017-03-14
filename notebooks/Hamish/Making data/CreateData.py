import pandas as pd

def CreateData(filepath="../../data/Raw Data.xls", drop_null=False, save=False):
    """
    Filepath: str or Pandas DataFrame
    """

    if isinstance(filepath, str):
        data = pd.read_excel(filepath)
    elif isinstance(filepath, pd.core.frame.DataFrame):
        data = filepath
    else:
        raise Exception("Use a Pandas DF or filepath str.")
    # Fix all dummy data
    # keep non-PSG data, ensure you keep zscore
    X = pd.concat((data.copy().iloc[:, :9], data.copy()['zscore']), axis=1)
    # choose just one out of two columns
    gender = pd.get_dummies(X.gender)['Male']
    # choose every column but the last dummy column, due to collinearity
    ethnicity = pd.get_dummies(X.ethnicity)
    no_allergies = pd.DataFrame(pd.get_dummies(X.allergies)['No'])
    no_allergies.columns = ['no_allergies']
    no_asthma = pd.DataFrame(pd.get_dummies(X.asthma)['No'])
    no_asthma.columns = ['no_asthma']
    premature = pd.DataFrame(X.term.astype(float) == 1.).astype(int)
    premature.columns = ['premature']
    no_gerd = pd.DataFrame(pd.get_dummies(X.gerd)['No'])
    no_gerd.columns = ['no_gerd']
    tonsilsize = pd.get_dummies(X.tonsilsize)
    tonsilsize.columns = ['tsize1', 'tsize2', 'tsize3', 'tsize4']
    binary_vars = (gender,
                   ethnicity,
                   no_allergies,
                   no_asthma,
                   premature,
                   no_gerd,
                   tonsilsize)
    new_X = pd.concat(binary_vars, axis=1)  # form final dataset
    numeric_vars = ['bmi', 'zscore', 'age', 'ahi']
    X = pd.concat((data[numeric_vars], new_X), axis=1)  # add in numeric data
    final_vars = ['ahi',
                  'bmi',
                  'zscore',
                  'age',
                  'premature',
                  'Male',
                  'Asian',
                  'Black',
                  'White',
                  'Hispanic',
                  'no_allergies',
                  'no_asthma',
                  'no_gerd',
                  'tsize1',
                  'tsize2',
                  'tsize3',
                  'tsize4']
    X = X[final_vars]
    # drop base categories for Kang et al. Points Model
    X = X.drop(["Asian", "tsize1"], axis=1)
    if drop_null:
        X = X.dropna()
    if save:
        X.to_csv("../../data/binarized_data.csv", index=False)
    y = X['ahi'] >= 5
    X.drop('ahi', axis=1, inplace=True)
    return X, y
