import pandas as pd

df = pd.read_csv('/scratch/en919/FINAL_DATA/reddit_sample_clean.csv')

df['subgroup'] = 'NO'

groups = {
    'Feminist': ['AskFeminists', 'Feminism', 'socialjustice101', 'GenderCritical', 'TwoXChromosomes', 'lgbt', 'ainbow'],
    'Black': ['BlackPeopleTwitter', 'Blackfellas', 'blackladies'],
    'Anti-SJW': ['CringeAnarchy', 'SocialJusticeInAction', 'sjwhate', 'KotakuInAction', '4chan', 'whiteknighting'],
    'Altright': ['WhiteRights', 'PussyPass', 'AntiPOZi', 'DebateAltRight', 'altright'],
    '/r/CoonTown': ['CoonTown'],
    '/r/uncensorednews': ['uncensorednews'],
    'Anti-altright': ['Fuckthealtright', 'AgainstHateSubreddits', 'onguardforthee'],
    'Anti-feminist': ['TheRedPill', 'MGTOW'],
    'Religion': ['islam', 'Judaism', 'Christianity'],
    'Misc': ['AskReddit', 'funny', 'food', 'AnimalsBeingJerks', 'gaming', 'EarthPorn', 'Fitness', 'science', 'books', 'movies', 'Music', 'anime', 'news', 'technology'],
    'Sport': ['sports', 'soccer', 'baseball'],
    'Conspiracy': ['WikiLeaks', 'conspiracy', 'HillaryForPrison'],
    '/r/The_Donald': ['The_Donald'],
    'Liberal': ['SandersForPresident', 'hillaryclinton', 'Liberal', 'democrats'],
    'Anti-Trump': ['esist', 'politics', 'EnoughTrumpSpam'],
    'Conservative': ['Conservative', 'Republican', 'AskTrumpSupporters', 'AskThe_Donald']
}

for group, subreddits in groups.items():
    df.loc[df.subreddit.isin(subreddits), 'subgroup'] = group

df = df.loc[df.subgroup != 'NO', :]


def train_test(df, col='subreddit', sample_size=3000, random_state=24):
    df_train = df.copy()
    df_train.reset_index(inplace=True)
    df_test = df_train.groupby(col).apply(
        lambda x: x.sample(sample_size, random_state=random_state))
    df_train = df_train.loc[~df_train['index'].isin(df_test['index'])].copy()
    df_train.drop('index', axis=1, inplace=True)
    df_test.drop('index', axis=1, inplace=True)
    return df_train, df_test


subreddits = df.subreddit.unique()

df = df.sort_values(by='score_ratio', ascending=False)
dfs = []
for subreddit in subreddits:
    df_sub = df.loc[df.subreddit == subreddit].copy()
    if subreddit in ['lgbt', 'CoonTown', 'uncensorednews', 'PoliticalDiscussion', 'The_Donald']:
        dfs.append(df_sub.head(300000))
    elif subreddit in ['MGTOW', 'TheRedPill', 'Fuckthealtright', 'conspiracy']:
        dfs.append(df_sub.head(150000))
    elif subreddit in ['BlackPeopleTwitter', 'CringeAnarchy']:
        dfs.append(df_sub.head(50000))
    else:
        dfs.append(df_sub.head(100000))

df = pd.concat(dfs)

rem_words = ['reddit', 'subreddit', 'subreddits',
             't_d', 'donald', 'trump', 'hillary', 'clinton']


def remove_words(s, rem_words):
    s = s.split()
    s = [word for word in s if word not in rem_words]
    return " ".join(s)


df.clean_version = df.clean_version.apply(remove_words, rem_words=rem_words)

df_train, df_test = train_test(df, col='subgroup', sample_size=20000)

subgroups = df.subgroup.unique()
dfs = []
for subgroup in subgroups:
    df_sub = df.loc[df.subgroup == subgroup].copy()
    if df_sub.shape[0] >= 250000:
        dfs.append(df_sub.sample(250000, random_state=24))
    else:
        dfs.extend([df_sub, df_sub.sample(
            250000 - df_sub.shape[0], random_state=24)])
df = pd.concat(dfs)

dfs = []
for subgroup in subgroups:
    df_sub = df_train.loc[df_train.subgroup == subgroup, :].copy()
    if df_sub.shape[0] >= 230000:
        dfs.append(df_sub.sample(230000, random_state=24))
    else:
        dfs.extend([df_sub, df_sub.sample(
            230000 - df_sub.shape[0], random_state=24)])
df_train = pd.concat(dfs)

df['labeled_data'] = '__label__' + df['subgroup'] + ' ' + df['clean_version']
df_train['labeled_data'] = '__label__' + \
    df_train['subgroup'] + ' ' + df_train['clean_version']
df_test['labeled_data'] = '__label__' + \
    df_test['subgroup'] + ' ' + df_test['clean_version']
df = df.sample(frac=1.0, random_state=24)
df_train = df_train.sample(frac=1.0, random_state=24)

df.labeled_data.to_csv(
    '/scratch/en919/FINAL_DATA/data_all_grouped.txt', encoding='utf-8', index=False)
df_train.labeled_data.to_csv(
    '/scratch/en919/FINAL_DATA/data_train_grouped.txt', encoding='utf-8', index=False)
df_test.labeled_data.to_csv(
    '/scratch/en919/FINAL_DATA/data_test_grouped.txt', encoding='utf-8', index=False)
