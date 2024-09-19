import pickle
from os import listdir
from os.path import isfile, join
from ride import common
from ride.common import CityResult

def load_data(mypath: str) -> dict:
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    r: dict[str: CityResult] = {}
    for name in onlyfiles:
        with open(join(mypath, name), 'rb') as f:
            r[name] = pickle.load(f)
            f.close()
    return r