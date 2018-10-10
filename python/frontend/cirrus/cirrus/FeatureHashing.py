# Apply feature hashing to specified columns

import json
import time
import boto3
from collections import deque
from utils import get_all_keys
from LambdaThread import LambdaThread
from threading import Thread

MAX_LAMBDAS = 400


class HashingThread(LambdaThread):
    def __init__(self, s3_bucket_input, s3_bucket_output, s3_key, columns, N):
        Thread.__init__(self)
        self.lamdba_dict = {
            "s3_bucket_input": s3_bucket_input,
            "s3_bucket_output": s3_bucket_output,
            "s3_key": s3_key,
            "action": "FEATURE_HASHING",
            "columns": columns,
            "N": N,
            "use_redis": "1"
        }


def FeatureHashing(s3_bucket_input, s3_bucket_output, columns, N, objects=[]):
    # Take a list of integer values (column indices) to perform the feature hashing with, for N buckets.
    s3_resource = boto3.resource("s3")
    if len(objects) == 0:
        # Allow user to specify objects, or otherwise get all objects.
        objects = get_all_keys(s3_bucket_input)
    # Hash the appropriate columns for each chunk
    start_hash = time.time()
    l_client = boto3.client("lambda")
    threads = deque()
    for i in objects:
        while len(threads) > MAX_LAMBDAS:
            t = threads.popleft()
            t.join()
        l = HashingThread(s3_bucket_input, s3_bucket_output, i, columns, N)
        l.start()
        threads.append(l)

    for t in threads:
        t.join()

    print("Feature hashing took {0} s".format(time.time() - start_hash))