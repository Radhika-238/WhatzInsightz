# WhatzInsightz – WhatsApp Chat Analyzer

An interactive WhatsApp chat analysis application that transforms exported conversations into meaningful insights using data analytics, visualization, and Natural Language Processing (NLP).

The application helps users explore messaging patterns, participant activity, word usage, sentiment, and conversation trends through a Streamlit dashboard.

**Live Application:** [WhatzInsightz](https://whatzinsightz.streamlit.app/)

## Project Overview

WhatsApp conversations contain valuable information about communication habits, engagement, and interaction patterns. However, analyzing large chat histories manually can be difficult.

WhatzInsightz simplifies this process by allowing users to upload exported WhatsApp conversations and explore their statistics through interactive visualizations.

### Objectives

- Analyze individual and group WhatsApp conversations.
- Identify messaging trends and participant activity.
- Visualize communication patterns over time.
- Explore frequently used words and emojis.
- Perform sentiment analysis using NLP.
- Provide an accessible interface for chat analytics.

## Features

### 1. Message Statistics

Displays key conversation metrics, including:

- Total messages
- Total words
- Media messages
- Shared links

### 2. User Activity Analysis

- Identifies the most active participants in group conversations.
- Compares message contributions across users.
- Helps understand participation patterns.

### 3. Timeline Analysis

- Monthly messaging trends.
- Daily activity patterns.
- Visualization of conversation activity over time.

### 4. Activity Heatmaps

- Analyzes messaging activity across days and hours.
- Highlights periods of high and low activity.
- Helps identify communication patterns.

### 5. Word Frequency Analysis

- Identifies commonly used words.
- Processes chat text to extract meaningful terms.
- Supports understanding of conversation topics and vocabulary.

### 6. Emoji Analysis

- Identifies frequently used emojis.
- Analyzes emoji distribution.
- Provides insights into expression patterns.

### 7. Sentiment Analysis

Uses VADER sentiment analysis from NLTK to classify text into:

- Positive
- Negative
- Neutral

Sentiment analysis provides an approximate indication of conversational tone and may be less reliable for Hinglish, slang, sarcasm, and code-mixed messages.

### 8. Response Time Analysis

Analyzes response-time patterns between conversation participants to explore interaction behavior.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Web Framework | Streamlit |
| Data Processing | Pandas |
| Data Visualization | Matplotlib, Seaborn |
| Natural Language Processing | NLTK, VADER |
| Emoji Analysis | Emoji |
| URL Extraction | URLExtract |
| Development Tools | Git, GitHub |

## How It Works

### Step 1 — Export WhatsApp Chat

Open a WhatsApp conversation and export the chat as a `.txt` file.

### Step 2 — Upload the Chat

Open the WhatzInsightz application and upload the exported file.

### Step 3 — Data Preprocessing

The application processes the chat export to extract structured information such as:

- Message timestamps
- Senders
- Message content
- Dates and times

### Step 4 — Data Analysis

The processed data is used to calculate message statistics, activity trends, word frequencies, emoji usage, and sentiment.

### Step 5 — Visualization

The application displays the results through charts, tables, and interactive dashboard components.

## Application Workflow

```text
WhatsApp Chat Export (.txt)
            |
            v
       File Upload
            |
            v
     Data Preprocessing
            |
            v
   Structured Chat Data
            |
            v
   Statistical Analysis
     /      |       \
Activity  Text     Sentiment
Analysis  Analysis  Analysis
     \      |       /
            v
   Streamlit Dashboard
```

## Project Structure

```text
WhatzInsightz/
|
├── app.py
├── helper.py
├── main.py
├── preprocessor.py
├── stop_hinglish.txt
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

**File Descriptions**

| File | Purpose |
|---|---|
| `app.py` | Streamlit application interface |
| `helper.py` | Functions for chat statistics and analysis |
| `main.py` | Additional application or processing logic |
| `preprocessor.py` | WhatsApp chat parsing and preprocessing |
| `stop_hinglish.txt` | Stopwords used during text analysis |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_FOLDER
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**macOS/Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open locally, typically at:

`http://localhost:8501`

Alternatively, access the deployed version:

**[WhatzInsightz Live Demo](https://whatzinsightz.streamlit.app/)**

## Key Learnings

This project provided practical experience in:

- Processing and structuring unorganized text data.
- Working with timestamps and time-based analysis.
- Performing exploratory data analysis on conversational datasets.
- Creating visualizations using Matplotlib and Seaborn.
- Implementing text preprocessing techniques.
- Applying lexicon-based sentiment analysis.
- Extracting and analyzing emojis and URLs.
- Developing interactive dashboards using Streamlit.
- Deploying Python applications.

## Limitations

- Chat parsing depends on the exported WhatsApp message format.
- Sentiment analysis may not accurately interpret sarcasm, slang, or multilingual conversations.
- Response-time estimates may not always reflect direct replies.
- Deleted messages and unavailable media content cannot be fully analyzed from text exports.
- The application relies on information available in the uploaded chat file.

## Future Improvements

- Improve support for different WhatsApp export formats.
- Introduce more advanced multilingual sentiment analysis.
- Add topic modeling to identify frequently discussed subjects.
- Improve response-time analysis using conversation context.
- Expand interactive filtering and visualization options.
- Add additional NLP-based insights.

## Author

**Radhika Agarwal**

B.Tech — Computer Science and Engineering

**GitHub:** [Radhika-238](https://github.com/Radhika-238)

**Live Application:** [WhatzInsightz](https://whatzinsightz.streamlit.app/)

---

WhatzInsightz demonstrates the practical application of data analytics, visualization, and Natural Language Processing to conversational data through an interactive web application.
