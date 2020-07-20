import os

import numpy as np
import pandas as pd

path = 'data/cit-HepPh/cit-hepPh/raw'


def count():

    files = [os.path.join(path, 'cit-HepPh.txt'), os.path.join(path, 'cit-HepPh-dates.txt')]
    skiprows = [4, 1]

    cit = pd.read_csv(files[0], sep='\t', header=None, skiprows=skiprows[0], dtype=str).values
    date = pd.read_csv(files[1], sep='\t', header=None, skiprows=skiprows[1], dtype=str)
    arr_date = date.values
    date.columns = ['id', 'date']
    date.set_index(['id'], inplace=True)

    ids = list()
    for i in range(arr_date.shape[0]):
        ids.append(arr_date[i][0])

    zeros = np.zeros((arr_date.shape[0], 10), dtype=np.int)
    df = pd.DataFrame(zeros)
    df.columns = ['1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001']
    df['id'] = ids
    df.set_index(['id'], inplace=True)

    for i in range(cit.shape[0]):

        if i % 10000 == 0:
            # break
            print(i)

        source = cit[i][0]
        cite = cit[i][1]

        # clean data
        ls = len(source)
        lc = len(cite)
        if ls < 7:
            source = '0' * (7-ls) + source
        if lc < 7:
            cite = '0' * (7-lc) + cite
        if source not in df.index or cite not in df.index:
            continue
        # end clean

        j = date.loc[source, 'date'][:4]
        if not j == '2002':
            df.loc[cite, j] = df.loc[cite, j] + 1

    df.to_csv(os.path.join(path, 'cit-count2.csv'))


def clean():

    # df = pd.read_csv(os.path.join(path, 'cit-count.csv'), dtype=str, header=0, index_col=None)
    # df.set_index(['id'], inplace=True)
    # print(df.shape)

    df = pd.read_csv(os.path.join(path, 'cit-count.csv'), dtype=str, header=0)

    print(df.shape)
    for index, row in df.iterrows():
        if row['id'][:2] == '11' or row['id'][:2] == '02':
            df.drop(index=[index], inplace=True)

    df.to_csv(os.path.join(path, 'cit-clean.csv'), index=None)


if __name__ == '__main__':

    # count()

    clean()

