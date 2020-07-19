import os

import numpy as np
import pandas




def main():

    path = 'data/cit-HepPh/cit-hepPh/raw'
    files = [os.path.join(path, 'cit-HepPh.txt'), os.path.join(path, 'cit-HepPh-dates.txt')]
    skiprows = [4, 1]

    cit = pandas.read_csv(files[0], sep='\t', header=None, skiprows=skiprows[0], dtype=str).values
    date = pandas.read_csv(files[1], sep='\t', header=None, skiprows=skiprows[1], dtype=str)
    arr_date = date.values

    date.columns = ['id', 'date']
    date.set_index(['id'], inplace=True)

    res = np.zeros((arr_date.shape[0], 11))
    for i in range(arr_date.shape[0]):
        res[i][0] = int(arr_date[i][0])

    # dict_date = date.to_dict()
    res = pandas.DataFrame(res)
    date.columns = ['id', '92', '93', '94', '95', '96', '97', '98', '99', '00', '01']

    res = pandas.DataFrame(res, columns=['id', '92', '93', '94', '95', '96', '97', '98', '99', '00', '01'], index=['id'])

    # for i in range(cit.shape[0]):
    #     source = cit[i][0]
    #     cite = cit[i][1]
    #     j = dict_date[source]
    #     j = dict_date[source][:4]
    #     if j is not '2002':
    #         j = int(j) - 1991
    #
    #         res[cite][j] += 1

    print(res.shape)


if __name__ == '__main__':
    main()

