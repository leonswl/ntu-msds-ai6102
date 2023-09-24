# first line: 1
@mem.cache
def get_data(path):
    data = load_svmlight_file(path)
    return data[0], data[1]
