from collections import Counter
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

def fetch_stats(selected_user, df):
    if selected_user != 'overall':
        df= df[df['user']==selected_user]
    #fetch no. of messages
    num_messages= df.shape[0]
    # fetch number of words
    words=[]
    for message in df['message']:
        words.extend(message.split())
    # fetch no of media messages
    media_count = df['message'].apply(lambda x: ('IMG' in x) or ('<Media omitted>' in x)).sum()
    #fetch no of links shared
    from urlextract import URLExtract
    links=[]
    extractor= URLExtract()
    for message in df['message']:
        links.extend(extractor.find_urls(message))
    # fetch stickers
    sticker_count = df['message'].apply(lambda x: 'STK' in x).sum()
    #return everything
    return num_messages, len(words), media_count, len(links), sticker_count


#most busy users
def most_busy_users(df):
    df = df[df['user'] != 'group_notification']
    x= df['user'].value_counts()
    percent_df=((df['user'].value_counts()/ df.shape[0]) * 100).round(2).reset_index()
    percent_df.columns= ['user', 'percent']
    return x, percent_df


#most common words
def most_common_words(selected_user, df):
    with open ('stop_hinglish.txt', 'r') as f:
        stop_words= f.read()
    if selected_user != 'overall':
        df= df[df['user']== selected_user]
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message']!= '<Media omitted>\n']

    words=[]
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words and word != '(file' and word != 'attached)':
                words.append(word)
    new_df= pd.DataFrame(Counter(words).most_common(20))
    new_df.columns= ['words', 'count']
    return new_df


# count of each emoji
import emoji
def emoji_count(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    emojis = []
    for message in df['message']:
        emojis_list= emoji.emoji_list(message)
        for item in emojis_list:
            emojis.append(item['emoji'])
    emoji_counts= Counter(emojis)
    emoji_df= pd.DataFrame(emoji_counts.items(), columns= ['emoji', 'count'])
    emoji_df= emoji_df.sort_values(by='count', ascending=False)
    return emoji_df


#display the busy months yearwise
def monthly_timeline(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    timeline= df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time=[]
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + '-' + str(timeline['year'][i]))
    timeline['time']= time
    return timeline


#display daily timeline
def daily_timeline(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    timeline= df.groupby('dates').count()['message'].reset_index()
    return timeline



#which day of the week is the busiest
def most_busy_day(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    busy_week_day= df['day_name'].value_counts()
    return busy_week_day


#which month of the year is the busiest
def most_busy_month(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    busy_month= df['month'].value_counts()
    return busy_month


#most busy hours
def most_busy_hours(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    hours_heatmap=  df.pivot_table(index= 'day_name', columns= 'period', values= 'message', aggfunc='count').fillna(0)
    return hours_heatmap




#who starts the conversation
def starts_conversation(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    df= df.sort_values(by='date')
    df['time_diff'] = df['date'].diff()
    threshold= pd.Timedelta(minutes= 30)
    df['starter']= df['time_diff'] >threshold
    starter_counts= df[df['starter']].groupby('user').size().sort_values(ascending=False)
    return starter_counts


#average response time
def avg_response_time(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    df= df.sort_values(by= 'date')
    df['prev_user']= df['user'].shift()
    df= df[df['user'] != df['prev_user']]
    df['time_diff'] = df['date'].diff().dt.total_seconds()
    df= df[df['time_diff'] > 0]
    response_time= (df.groupby('user')['time_diff'].mean().sort_values())
    return response_time


#sentiment analysis
def sentiment_analysis (selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    vader= SentimentIntensityAnalyzer()
    sentiments= []
    for message in df['message']:
        if message:
            score= vader.polarity_scores(message)['compound']
        else:
            score=0
        if score > 0:
            sentiments.append('positive')
        elif score < 0:
            sentiments.append('negative')
        else:
            sentiments.append('neutral')
    df['sentiments'] = sentiments
    sentiments= df['sentiments'].value_counts()
    return sentiments




