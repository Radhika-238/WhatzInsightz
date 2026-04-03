# WhatzInsightz – WhatsApp Chat Analyzer

WhatzInsightz is a data-driven web application that analyzes WhatsApp chat exports and generates meaningful insights using data visualization and natural language processing.

## Overview

This project allows users to upload WhatsApp chat data and explore communication patterns, user activity, and sentiment. It is designed to provide analytical insights through an intuitive and interactive interface built with Streamlit.

## Features

- Message statistics including total messages, words, media, and links
- Identification of most active users in a conversation
- Monthly and daily activity timelines
- Heatmap visualization of chat activity by day and time
- Most frequently used words analysis
- Emoji usage analysis
- Sentiment analysis using VADER (Positive, Negative, Neutral)
- Response time analysis between users

## Tech Stack

- Python
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- NLTK (VADER Sentiment Analysis)
- Emoji
- URLExtract

## How It Works

1. Export a WhatsApp chat as a `.txt` file
2. Upload the file through the application interface
3. The app processes and visualizes insights instantly

## Project Structure

├── app.py
├── helper.py
├── preprocessor.py
├── stop_hinglish.txt
├── requirements.txt
