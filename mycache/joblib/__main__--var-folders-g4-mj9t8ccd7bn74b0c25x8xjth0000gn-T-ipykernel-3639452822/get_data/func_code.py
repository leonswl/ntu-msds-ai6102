# first line: 1
@mem.cache
def get_data():
    data = load_svmlight_files(Path("artifacts/"))
    return data[0], data[1]
