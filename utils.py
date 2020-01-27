import numpy as np


def field_name_changer(df, columns_name_list):
    for value in columns_name_list[:-1]:
        df[value] = df[value].apply(lambda x: str(x) + value)


def bucketizer(df, columns_name_list):
    for value in columns_name_list[:-1]:
        column_max = df[value].max()
        column_min = df[value].min()
        stddev = df[value].std()

        number_of_buckets = 1
        if stddev != 0:
            number_of_buckets = ((column_max - column_min) // (stddev))

        buckets = np.arange(number_of_buckets, dtype=np.float).tolist()
        # TODO add the binning
def binning(df, columns_name_list, quantiles_num):
    for value in columns_name_list[:-1]:
        if df[value].nunique() == 1:
            df.drop([value], axis=1)
        else:
            df[value] = pd.qcut(df[value], q=quantiles_num, duplicates='drop')

def word_frequencies(df, columns_name_list):
    for value in columns_name_list[:-1]:
        df[value] = df[value].str.split(expand=True).stack().value_counts()