import streamlit as st

import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(
    page_title="WhatzInsightz",
    layout="wide"
)

st.title('WhatsApp Chat Analyzer')
st.text('Open sidebar to upload files')


#upload the file
uploaded_file= st.sidebar.file_uploader('Choose conversation to be analyzed')
if uploaded_file:
    bytes_data= uploaded_file.getvalue()
    data= bytes_data.decode('utf-8')

    st.header('your messages sample')
    df= preprocessor.preprocess(data)
    st.table(df[['user', 'message']].head(), hide_index=True)


    #fetch unique users
    user_list= df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, 'overall')
    selected_user= st.sidebar.selectbox('Select User', user_list)

    if st.sidebar.button('Show Analysis'):
        st.title('top statistics')
        num_messages, words, media_count, links, stickers = helper.fetch_stats(selected_user, df)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader('Total Messages')
            st.text(num_messages)

        with col2:
            st.subheader('Total Words')
            st.text(words)

        with col3:
            st.subheader('Media Shared')
            st.text(media_count)

        with col4:
            st.subheader('urls shared')
            st.text(links)



        #finding the busiest users in group
        if selected_user == 'overall':
            st.header('most busy people')
            x, new_df= helper.most_busy_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='pink')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.table(new_df.head(), hide_index=True)


        #world cloud


        #most common words
        st.header('most common words')
        common_words= helper.most_common_words(selected_user, df)
        col1, col2= st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.barh(common_words['words'], common_words['count'], color='lightpink')
            plt.xticks(rotation= 'vertical')
            st.pyplot(fig)
        with col2:
            st.table(common_words.head(), hide_index=True)


        #most common emojis
        st.header('most common emojis')
        emoji_df = helper.emoji_count(selected_user, df)
        col1, col2= st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.pie(emoji_df['count'].head(), labels= emoji_df['emoji'].head(), autopct= '%.2f' )
            st.pyplot(fig)
        with col2:
            st.table(emoji_df.head(), hide_index=True)


        #monthly timeline
        st.header('monthly timeline')
        timeline_df= helper.monthly_timeline(selected_user, df)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.plot(timeline_df['time'], timeline_df['message'])
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            timeline_df= timeline_df.sort_values(by= 'message', ascending = False)
            st.table(timeline_df[['time' , 'message']].head(), hide_index= True)


        #daily timeline
        st.header('daily timeline')
        timeline_df = helper.daily_timeline(selected_user, df)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.plot(timeline_df['dates'], timeline_df['message'])
            plt.xticks(rotation='vertical')
            plt.figure(figsize=(10, 10))
            st.pyplot(fig)
        with col2:
            timeline_df = timeline_df.sort_values(by='message', ascending=False)
            st.table(timeline_df[['dates', 'message']].head(), hide_index= True)


        #most busy weekdays
        st.header('most busy week days')
        busy_week_day= helper.most_busy_day(selected_user, df)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.bar(busy_week_day.index, busy_week_day.values)
            st.pyplot(fig)
        with col2:
            st.table(busy_week_day)


        #most busy month
        st.header('most busy month')
        busy_month = helper.most_busy_month(selected_user, df)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.table(busy_month)


        #most busy hours
        st.header('most busy hours')
        busy_hours = helper.most_busy_hours(selected_user, df)
        fig, ax = plt.subplots(figsize=(2,2))
        ax= sns.heatmap(busy_hours, cmap='coolwarm', linewidth= 0.5, linecolor= 'white', ax=ax  )
        ax.set_xlabel('hours', fontsize= 3)
        ax.set_ylabel('weekdays', fontsize=3)
        ax.tick_params(axis='x', labelsize=3)
        ax.tick_params(axis='y', labelsize=3)
        st.pyplot(fig)
        
        # ===============================
        # WHO STARTS + RESPONSE TIME
        # ===============================
        
        col1, col2, col3 = st.columns(3)
        
        # 🔹 Who starts conversation
        with col1:
            st.header('Who starts the conversation')
            starter_counts = helper.starts_conversation(selected_user, df)
            st.dataframe(starter_counts)
        
        # 🔹 Response insights
        response_time = helper.avg_response_time(selected_user, df)
        
        if not response_time.empty:
            response_time = response_time / 60  # convert to minutes
        
            with col2:
                st.header('Responses')
        
                fastest = response_time.idxmin()
                slowest = response_time.idxmax()
        
                st.subheader(f'Fastest responder: {fastest}')
                st.subheader(f'Slowest responder: {slowest}')
        
            with col3:
                st.header('Average Response Time (minutes)')
        
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(response_time.index, response_time.values)
        
                ax.set_ylabel("Minutes")
                plt.xticks(rotation=45)
        
                st.pyplot(fig, use_container_width=True)
        
        else:
            st.warning("Not enough data for response time analysis")
        
        
        # ===============================
        # SENTIMENT ANALYSIS
        # ===============================
        
        st.header('Sentiment Analysis')
        
        sentiments = helper.sentiment_analysis(selected_user, df)
        
        col1, col2 = st.columns(2)
        
        # 🔹 Bar chart
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(sentiments.index, sentiments.values)
        
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
        
            st.pyplot(fig, use_container_width=True)
        
        # 🔹 Table
        with col2:
            st.dataframe(sentiments)
