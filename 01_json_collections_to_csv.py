# adding paths for user libraries
import bz2
import glob
import os
import sys

import pandas as pd

from pysmap import SmappCollection

# get command line arguments
args = sys.argv

# get lists of archived and non-archived files
files_bz = glob.glob(
    '/scratch/olympus/us_election_{}_2016/data/*.json.bz2'.format(args[1]))
files_json = glob.glob(
    '/scratch/olympus/us_election_{}_2016/data/*.json'.format(args[1]))
dirpath = "/scratch/en919/"

# read list of parsed files
if os.path.isfile('files_{}.txt'.format(args[1])):
    with open('files_{}.txt'.format(args[1]), 'r') as f:
        parsed_files = f.read().splitlines()
else:
    parsed_files = []


def dump_tweets(filename, retweets, fields=None):
    collection = SmappCollection('json', filename)
    collection = collection.user_language_is('en')
    if fields is None:
        fields = ['id', 'text', 'timestamp_ms', 'user.id_str']

    if retweets:
        collection = collection.get_retweets()
        collection.dump_to_csv('/scratch/en919/retw_' +
                               args[1] + '.csv', fields)
    else:
        collection = collection.exclude_retweets()
        collection.dump_to_csv(
            '/scratch/en919/no_retw_' + args[1] + '.csv', fields)


def split_csv(retw):
    if retw:
        df = pd.read_csv('/scratch/en919/retw_{}.csv'.format(args[1]))
        df.columns = ['id', 'text', 'timestamp_ms', 'user_id']
        df['retweet'] = 1
    else:
        df = pd.read_csv('/scratch/en919/no_retw_{}.csv'.format(args[1]))
        df.columns = ['id', 'text', 'timestamp_ms', 'user_id']
        df['retweet'] = 0
    df = df.loc[~df.timestamp_ms.isin(['timestamp_ms'])]
    df.timestamp_ms = pd.to_numeric(df.timestamp_ms) / 1000
    df['day'] = pd.to_datetime(df.timestamp_ms, unit='s')
    df['yr_mnth'] = df['day'].apply(
        lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else '')
    dates = sorted(list(df.yr_mnth.unique()))
    df['text'] = df['text'].map(
        lambda x: x.encode('unicode-escape').decode('utf-8'))
    for date in dates:
        df_date = df.loc[df.yr_mnth == date].copy()
        new_file = '/scratch/en919/tweets_csv_new/{}/{}.csv'.format(
            args[1], date)
        if os.path.isfile(new_file):
            with open(new_file, 'a') as f:
                df_date.to_csv(f, header=False, encoding='utf-8')
        else:
            df_date.to_csv(new_file, index=False, encoding='utf-8')


for filename in files_bz:
    short_name = os.path.basename(filename)
    if short_name in parsed_files:
        continue
    newfilepath = os.path.join(dirpath, short_name + '.decompressed')
    with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filename, 'rb') as f:
        for data in iter(lambda: f.read(200000 * 1024), b''):
            new_file.write(data)
    dump_tweets(newfilepath, False)
    dump_tweets(newfilepath, True)

    split_csv(retw=False)
    split_csv(retw=True)
    os.remove('/scratch/en919/retw_{}.csv'.format(args[1]))
    os.remove('/scratch/en919/no_retw_.csv'.format(args[1]))
    os.remove(newfilepath)
    with open('files_{}.txt'.format(args[1]), 'a') as f:
        f.write(short_name + '\n')

for filename in files_json:
    short_name = os.path.basename(filename)
    if short_name in parsed_files:
        continue
    dump_tweets(filename, False)
    dump_tweets(filename, True)

    split_csv(retw=False)
    split_csv(retw=True)

    os.remove('/scratch/en919/retw_{}.csv'.format(args[1]))
    os.remove('/scratch/en919/no_retw_.csv'.format(args[1]))

    with open('files_{}.txt'.format(args[1]), 'a') as f:
        f.write(short_name + '\n')
