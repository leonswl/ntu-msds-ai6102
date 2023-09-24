# first line: 1
@mem.cache
def get_data(path):
    data = load_svmlight_file(
        f=path,
        n_features=123)
    return data[0], data[1]
