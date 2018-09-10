# import required libraries
import glob
import string

import pandas as pd

from redditscore.tokenizer import CrazyTokenizer

tokenizer = CrazyTokenizer(remove_punct=True, twitter_handles='',
                           hashtags='', urls='', ignore_stopwords='english',
                           subreddits='', reddit_usernames='', keepcaps=False,
                           ignore_quotes=True, numbers='')

# get the list of all csv files
files = sorted(glob.glob('/scratch/en919/reddit/september_2017/*.gz'))

# read the files
df = pd.concat((pd.read_csv(f, compression='gzip', lineterminator='\n')
                for f in files))


def isstr(s):
    return isinstance(s, str)


def clean_df(data):
    df = data.copy()
    df = df.loc[df.body.apply(isstr)]
    df.body = df.body.str.replace('&gt.+\n', ' ')
    df.replace({r'\r\n': ''}, regex=True, inplace=True)
    df.replace({r'\n': ''}, regex=True, inplace=True)
    df.replace({r'\r': ''}, regex=True, inplace=True)
    df.body = df.body.str.strip()
    df = df.loc[df.body.apply(isstr)]
    df = df.loc[(df.subreddit != 'ShitRedditSays') |
                (~df.body.str.contains('replied'))]
    df = df.loc[~df.body.str.contains("I'm a bot.", regex=False)]
    df = df.sort_values(by='created_utc')
    df = df.loc[df.score >= 0, :]
    df = df.loc[~df.author.isin(['TrumpTrain-bot', 'trumpcoatbot', 'DJ_Spam'])]
    df = df.loc[df.body != '[^bot ^info](/r/youtubefactsbot/wiki/index)']

    df = df.loc[df.body.apply(isstr)]
    df = df.loc[df.body.notnull()]

    return df


df = clean_df(df)

g = df.groupby('link_id').size().reset_index()
g.columns = ['link_id', 'num_comments']
df = pd.merge(df, g, on='link_id')
df['score_ratio'] = df['score'] / df['num_comments']
df = df.sort_values(by='score_ratio', ascending=False)
df = df.groupby('subreddit').head(300000)

# get lists of subreddits
subreddits = list(df.subreddit.unique())


def clean_post(s, tokenizer=tokenizer):
    # function to clean comments
    try:
        s = s.encode('utf-8').decode('unicode-escape')
        s = ''.join(filter(lambda x: x in string.printable, s))
    except UnicodeDecodeError:
        pass
    s = tokenizer.tokenize(s)
    s = [word for word in s if len(word) > 1]
    return " ".join(s)


# for tokenizing URLs
df.body = df.body.str.replace('\(http', ' http')
df.body = df.body.str.replace(']http', ' http')
# remove everything in squared brackets
df.body = df.body.str.replace(r"\[.*\]", " ")
# clean comments
df['clean_version'] = df.body.apply(clean_post)

# remove more bad comments
df.clean_version = ' ' + df.clean_version + ' '
remove_comments = ['auto-archiving', 'auto-archive', 'thank participating however',
                   'post removed', 'defaulted one day messaging', 'TIME utc', 'archive org', 'nbsp', 'sister sub']
for comment in remove_comments:
    df = df.loc[~df.clean_version.str.contains(comment)]
df.clean_version = df.clean_version.str.replace(
    ' ahs | bpt | http | ets | :/ | rp | 2x | www | trp | t_d ', ' ')
df.clean_version = df.clean_version.str.replace(
    ' ahs | bpt | http | ets | :/ | rp | 2x | www | trp | t_d ', ' ')
df.clean_version = df.clean_version.str.strip()
df = df.loc[df.clean_version.notnull()]

for subreddit in subreddits:
    if subreddit in ['DebateAltRight', 'EarthPorn',  'NeutralPolitics', 'esist', 'ShitRedditSays', 'ainbow', 'AskReddit', 'TheRedPill', 'CoonTown', 'uncensorednews', 'AskTrumpSupporters',
                     'AskFeminists', 'changemyview', 'GenderCritical', 'PoliticalDiscussion', 'SandersForPresident', 'TwoXChromosomes', 'KotakuInAction', 'The_Donald', 'HillaryForPrison', 'AnimalsBeingJerks',
                     'CringeAnarchy', 'dankmemes', 'TumblrInAction', 'Fuckthealtright', 'AgainstHateSubreddits', 'AskThe_Donald', 'BlackPeopleTwitter', 'hillaryclinton', 'EnoughTrumpSpam', 'sjwhate',
                     'Blackfellas', 'blackladies', 'PussyPass', 'AntiPOZi', 'SocialJusticeInAction', 'moderatepolitics', 'onguardforthee', 'WhiteRights', 'whiteknighting', 'uspolitics', 'socialjustice101']:
        df.clean_version = df.clean_version.str.replace(subreddit.lower(), '')


df = df.loc[df.clean_version.apply(isstr)]

df.to_csv('/scratch/en919/FINAL_DATA/reddit_sample_clean.csv')
