from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
#import boto3
from os import listdir
from os.path import isfile, join
#import fastparquet as fp

def read_process(filname, sep="\t"):
    col_names = ["user", "item", "rate", "st"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
    df["user"] -= 1
    df["item"] -= 1
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["rate"] = df["rate"].astype(np.float32)
    return df


# def read_process2(ndownload):
#     s3 = boto3.resource('s3')
#     mybucket = s3.Bucket('grp-ds-users')
#     for obj in mybucket.objects.filter(Prefix='james.thomson/adp/svd/days_28/annotations_matrix/'):
#         print('{0}:{1}'.format(mybucket.name, obj.key))
#     objects = [obj for obj in mybucket.objects.filter(Prefix='james.thomson/adp/svd/days_28/annotations_matrix/')]
#     print(objects[:2])
#     for i,obj in enumerate(objects[1:ndownload]):
#         print(i, obj.key)
#         print(type(obj))
#         s3.meta.client.download_file(mybucket.name,obj.key,'/tmp/'+str(i)+'.parquet')
#     pass

# def read_process3(ndownload):
#     s3=boto3.client('s3')
#     lst=s3.list_objects(Bucket='grp-ds-users',
#         Prefix='grp-ds-users:james.thomson/adp/svd/days_28/annotations_matrix/')['Contents']
#     print(lst)
#     for s3_key in lst[:ndownload]:
#         s3_object = s3_key['Key']
#         if not s3_object.endswith("/"):
#             s3.download_file('grp-ds-users', s3_object, s3_object)
#         else:
#             if not os.path.exists(s3_object):
#                 os.makedirs(s3_object)

# def read_process4(N = 15):
    
#     onlyfiles = [f for f in listdir('data/') if isfile(join('data/', f))]
#     df = []
#     for f in onlyfiles[:N]:
#         print('loading:', f)
#         pfile = fp.ParquetFile('data/'+f)
#         df.append(pfile.to_pandas())
#     df = pd.concat(df,axis=0)

#     for col in ("userIndex", "annotationIndex"):
#          df[col] = df[col].astype(np.int32)
#     print(df.head())
#     #print(len(df))
#     print('sorting')

#     #uidDict = {i:x for i,x in enumerate(list(set(df['userIndex'])))}
#     #iidDict = {i:x for i,x in enumerate(list(set(df['annotationIndex'])))}
#     print('making Dicts:')
#     uidDictRev = {x:i for i,x in enumerate(list(set(df['userIndex'])))}
#     iidDictRev = {x:i for i,x in enumerate(list(set(df['annotationIndex'])))}

#     print('reindexing datasets:')
#     print('annotations...')
#     df['annotationIndex'] = [iidDictRev[x] for x in df['annotationIndex']]
#     print('users...')
#     df['userIndex'] = [uidDictRev[x] for x in df['userIndex']]
#     # print(len(set(df['annotationIndex'])))
#     # print(len(set(df['userIndex'])))
#     print('done sorting')

#     df["weight"] = df["weight"].astype(np.float32)
#     df['weight'] = df['weight'] - np.mean(df['weight'])
#     df['weight'] = df['weight']/np.std(df['weight'])


#     print(df.head())
#     return df


def read_process4(N = 15):
    
    onlyfiles = [f for f in listdir('data/') if isfile(join('data/', f))]
    df = []
    for f in onlyfiles[:N]:
        print('loading:', f)
        #pfile = fp.ParquetFile('data/'+f)
        tmp = pd.read_csv('data/'+f)
        tmp.columns = ['userIndex','annotationIndex','weight']
        df.append(tmp)
    df = pd.concat(df,axis=0)

    for col in ("userIndex", "annotationIndex"):
         df[col] = df[col].astype(np.int32)
    print(df.head())
    #print(len(df))
    print('sorting')

    #uidDict = {i:x for i,x in enumerate(list(set(df['userIndex'])))}
    #iidDict = {i:x for i,x in enumerate(list(set(df['annotationIndex'])))}
    print('making Dicts:')
    uidDictRev = {x:i for i,x in enumerate(list(set(df['userIndex'])))}
    iidDictRev = {x:i for i,x in enumerate(list(set(df['annotationIndex'])))}

    print('reindexing datasets:')
    print('annotations...')
    df['annotationIndex'] = [iidDictRev[x] for x in df['annotationIndex']]
    print('users...')
    df['userIndex'] = [uidDictRev[x] for x in df['userIndex']]
    # print(len(set(df['annotationIndex'])))
    # print(len(set(df['userIndex'])))
    print('done sorting')

    means = np.mean(df['weight'])
    sds = np.std(df['weight'])
    df["weight"] = df["weight"].astype(np.float32)
    df['weight'] = df['weight'] - means
    df['weight'] = df['weight']/sds


    print(df.head())
    return df, means,sds





class ShuffleIterator(object):
    """
    Randomly generate batches
    """
    def __init__(self, inputs, batch_size=10):
        self.inputs = inputs
        self.batch_size = batch_size
        self.num_cols = len(self.inputs)
        self.len = len(self.inputs[0])
        self.inputs = np.transpose(np.vstack([np.array(self.inputs[i]) for i in range(self.num_cols)]))

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        ids = np.random.randint(0, self.len, (self.batch_size,))
        out = self.inputs[ids, :]
        return [out[:, i] for i in range(self.num_cols)]


class OneEpochIterator(ShuffleIterator):
    """
    Sequentially generate one-epoch batches, typically for test data
    """
    def __init__(self, inputs, batch_size=10):
        super(OneEpochIterator, self).__init__(inputs, batch_size=batch_size)
        if batch_size > 0:
            self.idx_group = np.array_split(np.arange(self.len), np.ceil(self.len / batch_size))
        else:
            self.idx_group = [np.arange(self.len)]
        self.group_id = 0

    def next(self):
        if self.group_id >= len(self.idx_group):
            self.group_id = 0
            raise StopIteration
        out = self.inputs[self.idx_group[self.group_id], :]
        self.group_id += 1
        return [out[:, i] for i in range(self.num_cols)]

if __name__ == '__main__':
    df = read_process4(10)
    

    # pfile = fp.ParquetFile('data/part-00000-ff2b6db9-fde7-4454-9666-7513786ec927.snappy.parquet')
    # df = pfile.to_pandas()
    

    # for col in ("userIndex", "annotationIndex"):
    #      df[col] = df[col].astype(np.int32)

    # print('sorting')

    # uidDict = {i:x for i,x in enumerate(list(set(df['userIndex'])))}
    # iidDict = {i:x for i,x in enumerate(list(set(df['annotationIndex'])))}

    # uidDictRev = {x:i for i,x in enumerate(list(set(df['userIndex'])))}
    # iidDictRev = {x:i for i,x in enumerate(list(set(df['annotationIndex'])))}

    # print(uidDictRev[uidDict[0]])
    # print(iidDictRev[iidDict[0]])

    # df['annotationIndex'] = [iidDictRev[x] for x in df['annotationIndex']]
    # df['userIndex'] = [uidDictRev[x] for x in df['userIndex']]

    # print(df.head())
    # print(np.max(df['annotationIndex']))
    # print(np.max(df['userIndex']))

    # n = 0
    # print(len(iidDictRev.keys()))
    # print(len(uidDictRev.keys()))
    # print(len(iidDict.keys()))
    # print(len(uidDict.keys()))
    # for k,v in list(iidDictRev.items())[:10]:
    #     print(k,v)