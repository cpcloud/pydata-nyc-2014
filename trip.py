#!/usr/bin/env python

from __future__ import print_function

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


def time_by(d, grouper='passenger_count', reducer='trip_time_in_secs',
            function='sum'):
    expr = by(getattr(d, grouper), s=getattr(getattr(d, reducer), function)())
    times = []
    cpus = psutil.NUM_CPUS
    cores = [2 ** i for i in range(int(np.log2(cpus)) + 1)]
    for core_count in cores:
        p = mp.Pool(core_count)
        with timeit('cores: %d' % core_count, times):
            compute(expr, map=p.map)
        p.close()
    return np.array(times)


if __name__ == '__main__':
    groupers = 'passenger_count', 'medallion', 'hack_license'
    reducers = 'trip_time_in_secs', 'trip_distance'
    results = {}
    filename = os.path.join(os.path.dirname(__file__), os.pardir, 'bcolz',
                            'trip.bcolz')
    assert os.path.exists(filename), '%r does not exist' % filename
    trip = Data(filename)
    for grouper, reducer in it.product(groupers, reducers):
        print('%s, %s' % (grouper, reducer))
        results[(grouper, reducer)] = time_by(trip, grouper, reducer)
        print()
    result = pd.DataFrame(results)
    result.to_csv(os.path.join('results', 'trip_by_results.csv'))
