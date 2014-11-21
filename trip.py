#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import time
import numpy as np
import multiprocessing as mp
import itertools as it
import pandas as pd
import psutil
from blaze import Data, by, compute
from contextlib import contextmanager


@contextmanager
def timeit(name, result):
    assert isinstance(result, list)
    start = time.time()
    yield
    stop = time.time()
    diff = stop - start
    result.append(diff)
    print('%s took %.3g s' % (name, diff))


def cores():
    cpus = psutil.NUM_CPUS
    return [2 ** i for i in range(int(np.log2(cpus)) + 1)]


def time_by(d, grouper='passenger_count', reducer='trip_time_in_secs',
            function='sum'):
    expr = by(getattr(d, grouper), s=getattr(getattr(d, reducer), function)())
    times = []
    for core_count in cores():
        p = mp.Pool(core_count)
        with timeit('cores: %d' % core_count, times):
            compute(expr, map=p.map)
        p.close()
    return np.array(times)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('filename')
    return p.parse_args()


def read_results(filename, header=(0, 1), index_col=0, ncores=None):
    df = pd.read_csv(filename, header=list(header), index_col=index_col)
    df.columns.names = ['grouper', 'reducer']
    df.index = pd.Index(ncores or cores(), name='cores')
    return df.T


if __name__ == '__main__':
    groupers = 'passenger_count', 'medallion', 'hack_license'
    reducers = 'trip_time_in_secs', 'trip_distance'
    results = {}

    p = parse_args()
    filename = os.path.abspath(p.filename)
    assert os.path.exists(filename), '%r does not exist' % filename
    ncores = cores()[-1]
    bcolz = Data(filename)
    trips = [bcolz]
    if ncores == 32:
        trips.append(pd.DataFrame(bcolz.data[:]))
    for trip, grouper, reducer in it.product(trips, groupers, reducers):
        print('%s, %s' % (grouper, reducer))
        results[(grouper, reducer)] = time_by(trip, grouper, reducer)
        print()
    result = pd.DataFrame(results)
    result.to_csv(os.path.join('results',
                               'trip_by_results_%d.csv' % cores()[-1]))
