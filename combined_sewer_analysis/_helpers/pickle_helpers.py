import pickle


def read_pickle(fn):
    pkl_file = open(fn, 'rb')
    result = pickle.load(pkl_file)
    pkl_file.close()
    return result


def write_pickle(object, fn):
    output = open(fn, 'wb')
    pickle.dump(object, output)
    output.close()
