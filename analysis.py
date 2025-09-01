import streamlit as st
import pandas as pd
import pandas_ta as ta
import requests
import numpy as np
from openai import OpenAI
import plotly.express as px
from bs4 import BeautifulSoup
from ta_prompt import SUMMARY2
import re
import markdown2
import json
from news_analysis import get_news_sentiment_gathered_data
from news_prompt import system_prompt_html

# --- API Keys ---
API_KEY = st.secrets["MARKETSTACK_API_KEY"]
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_KEY)

# Optional ticker mapping
MARKETSTACK_TICKER_MAP = {}

def update_progress(progress_bar, stage, progress, message):
    progress_bar.progress(progress)
    st.text(message)
    st.empty()

# --- Fetch from Marketstack ---
def fetch_marketstack_data(ticker, period):
    ticker = MARKETSTACK_TICKER_MAP.get(ticker, ticker)

    period_map = {
        "3 Months": 65,
        "6 Months": 130,
        "1 Year": 260
    }
    limit = period_map.get(period, 260)

    url = f"http://api.marketstack.com/v2/tickers/{ticker}/eod?access_key={API_KEY}&limit={limit}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        json_response = response.json()
        eod_data = json_response.get("data", {}).get("eod", [])
        if not eod_data:
            return None

        df = pd.DataFrame(eod_data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        df = df[["open", "high", "low", "close", "volume", "symbol", "exchange"]]
        df.columns = [col.capitalize() for col in df.columns]

        # Resample weekly OHLCV
        
        return df
    except Exception as e:
        st.error(f"Marketstack API Error for {ticker}: {e}")
        return None

def bollingerbands(company_name, data_text):
    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an AI model designed to assist long-term day traders in analyzing stock market data. "
                    "Your primary task is to interpret stock trading data, especially focusing on Bollinger Bands, "
                    "to identify key market trends. When provided with relevant data you will: "
                    "Analyze the stock's current position relative to its Bollinger Bands (upper, middle, or lower bands) and provide insights."
            },
            {
                "role": "user",
                "content": f"Provide a concise analysis of {company_name} using the given stock data {data_text}, focusing only on insights from the Bollinger Bands indicator."
            },
        ]
    )
    response = chat_completion.choices[0].message.content
    return response
def SMA(company_name,data_text):
    
    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            # System message to define the assistant's behavior
            {
                "role": "system",
                "content":"You are an AI model designed to assist long-term day traders in analyzing stock market data."
                    "Your primary task is to interpret stock trading data, especially focusing on 20, 50, and 200 Simple Moving Averages (SMA),"
                    "to identify key market trends. When provided with relevant data you will:"
                    "\n- Analyze the stock's current position relative to its 20, 50, and 200 SMAs."
                    "\n- Assess if the stock is in an uptrend, downtrend, or nearing a breakout based on the relationships between the SMAs."
                    "\n- Determine if the stock is prone to a reversal by analyzing price movements, SMA crossovers, and the stock's position relative to key SMAs."
                    "\n- Provide a concise, expert-level explanation of your analysis, including how specific SMA characteristics (e.g., crossovers, price deviation from SMAs, trend strength)"
                    "indicate potential market moves."
                    "\n\nEnsure that your explanations are clear and easy to understand, even for users with little to no trading experience, avoiding complex jargon or offering simple definitions where necessary."
                    "Your output should balance depth and simplicity, offering actionable insights for traders while being accessible to non-traders."
                
            },
            # User message with a prompt requesting stock analysis for a specific company
            {
                "role": "user",
                "content": f"Provide a concise analysis of {company_name} using the given stock data {data_text}, focusing only on insights from the SMA indicator."

            },
        ]
    )

# Output the AI's response
    response = chat_completion.choices[0].message.content
    return response


def RSI(company_name,data_text):
    
    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            # System message to define the assistant's behavior
            {
                "role": "system",
                "content":"You are an AI model designed to assist long-term day traders in analyzing stock market data."
                    "Your primary task is to interpret stock trading data, especially focusing on the Relative Strength Index (RSI),"
                    "to identify key market trends. When provided with relevant data you will:"

                    "\n- Analyze the stock's current RSI values to determine if it is overbought, oversold, or in a neutral range."
                    "\n- Assess if the stock is in an uptrend, downtrend, or nearing a potential reversal based on RSI levels and patterns."
                    "\n- Determine if the stock is prone to a reversal by analyzing RSI divergences (bullish or bearish), overbought/oversold conditions, and the stock's momentum."
                    "\n- Provide a concise, expert-level explanation of your analysis, including how specific RSI characteristics (e.g., divergence, trend strength, threshold breaches)"
                    "indicate potential market moves."
                
            },
            # User message with a prompt requesting stock analysis for a specific company
            {
                "role": "user",
                "content": f"Provide a concise analysis of {company_name} using the given stock data {data_text}, focusing only on insights from the RSI indicator."

            },
        ]
    )

# Output the AI's response
    response = chat_completion.choices[0].message.content
    return response

def MACD(company_name,data_text):
    
    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            # System message to define the assistant's behavior
            {
                "role": "system",
                "content":"You are an AI model designed to assist long-term day traders in analyzing stock market data."
                    "Your primary task is to interpret stock trading data, especially focusing on the MACD (Moving Average Convergence Divergence), MACD Signal Line, and MACD Histogram,"
                    "to identify key market trends. When provided with relevant data you will:"
                    "\n- Analyze the stock's MACD line, Signal Line, and Histogram to assess trend strength and potential price direction."
                    "\n- Assess if the stock is in an uptrend, downtrend, or nearing a crossover by analyzing the MACD line relative to the Signal Line."
                    "\n- Determine if the stock is prone to a reversal by examining MACD crossovers, divergences, and changes in the MACD Histogram."
                    "\n- Provide a concise, expert-level explanation of your analysis, including how specific MACD characteristics (e.g., crossover points, divergence, histogram changes)"
                    "indicate potential market moves."
                    "\n\nEnsure that your explanations are clear and easy to understand, even for users with little to no trading experience, avoiding complex jargon or offering simple definitions where necessary."
                    "Your output should balance depth and simplicity, offering actionable insights for traders while being accessible to non-traders."
                
            },
            # User message with a prompt requesting stock analysis for a specific company
            {
                "role": "user",
                "content": f"Provide a concise analysis of {company_name} using the given stock data {data_text}, focusing only on insights from the MACD indicator."

            },
        ]
    )

# Output the AI's response
    response = chat_completion.choices[0].message.content
    return response


def OBV(company_name,data_text):
    
    chat_completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            # System message to define the assistant's behavior
            {
                "role": "system",
                "content":"You are an AI model designed to assist long-term day traders in analyzing stock market data."
                    "Your primary task is to interpret stock trading data, especially focusing on On-Balance Volume (OBV),"
                    "to identify key market trends. When provided with relevant data you will:"

                    "\n\n- Read and extract relevant data from PDF and CSV files."
                    "\n- Analyze the stock's OBV to assess the relationship between volume and price movement."
                    "\n- Assess if the stock is in an uptrend, downtrend, or nearing a breakout by evaluating OBV trends and volume momentum."
                    "\n- Determine if the stock is prone to a reversal by analyzing OBV divergences (where OBV moves in the opposite direction of price), which can signal potential trend changes."
                    "\n- Provide a concise, expert-level explanation of your analysis, including how specific OBV characteristics (e.g., divergence, volume spikes, confirmation of price moves)"
                    "indicate potential market moves."

                    "\n\nEnsure that your explanations are clear and easy to understand, even for users with little to no trading experience, avoiding complex jargon or offering simple definitions where necessary."
                    "Your output should balance depth and simplicity, offering actionable insights for traders while being accessible to non-traders."
                
            },
            # User message with a prompt requesting stock analysis for a specific company
            {
                "role": "user",
                "content": f"Provide a concise analysis of {company_name} using the given stock data {data_text}, focusing only on insights from the OBV indicator."

            },
        ]
    )

# Output the AI's response
    response = chat_completion.choices[0].message.content
    return response


def ADX(company_name,data_text):
    
    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            # System message to define the assistant's behavior
            {
                "role": "system",
                "content":"You are an AI model designed to assist long-term day traders in analyzing stock market data."
                    "Your primary task is to interpret stock trading data, especially focusing on the Average Directional Index (ADX),"
                    "to identify key market trends. When provided with relevant data you will:"

                    "\n- Analyze the stock's ADX values to assess the strength of the current trend, regardless of its direction."
                    "\n- Assess if the stock is in a strong or weak trend based on ADX levels, with particular attention to rising or falling ADX values."
                    "\n- Determine if the stock is prone to a trend reversal by analyzing ADX indicating whether the market is gaining or losing trend strength."
                    "\n- Provide a concise, expert-level explanation of your analysis, including how specific ADX characteristics (e.g., ADX crossovers, trend strength, or weakening trends)"
                    "indicate potential market moves."

                    "\n\nEnsure that your explanations are clear and easy to understand, even for users with little to no trading experience, avoiding complex jargon or offering simple definitions where necessary."
                    "Your output should balance depth and simplicity, offering actionable insights for traders while being accessible to non-traders."
                
            },
            # User message with a prompt requesting stock analysis for a specific company
            {
                "role": "user",
                "content": f"Provide a concise analysis of {company_name} using the given stock data {data_text}, focusing only on insights from the ADX indicator."
                
            },
        ]
    )

# Output the AI's response
    response = chat_completion.choices[0].message.content
    return response


# --- Technical indicators with weights ---
def calculate_technical_indicators(data, ticker, weight_choice=None):
    """
    Calculate various technical indicators, prepare them for AI analysis,
    and compute a weighted technical score.

    Args:
        data (pd.DataFrame): The input financial data.
        ticker (str): The stock ticker.
        weights (dict): Optional dict of weights for each indicator.

    Returns:
        Tuple: (results dict, recent_data, availability, scores, weighted_score)
    """
    short_term_weights = {
    "sma": 0.1,
    "rsi": 0.3,
    "macd": 0.3,
    "obv": 0.1,
    "adx": 0.1,
    "bbands": 0.1
    }
    long_term_weights = {
        "sma": 0.4,
        "rsi": 0.1,
        "macd": 0.15,
        "obv": 0.15,
        "adx": 0.2,
        "bbands": 0.0
    }

    weights = {
            "sma": 0.2,
            "rsi": 0.2,
            "macd": 0.2,
            "obv": 0.2,
            "adx": 0.2,
            "bbands": 0.0  # Set to 0 if not using
        }

# Choose the right weights
    if weight_choice == "Short Term":
        weights = short_term_weights
    if weight_choice == "Long Term":
        weights = long_term_weights
    if weight_choice == "Default":
        weights = weights

    # --- Default Weights if not provided ---

    # Initialize availability flags
    sma_available = False
    rsi_available = False
    macd_available = False
    obv_available = False
    adx_available = False
    bbands_available = False

    # --- Calculate SMA ---
    if 'Close' in data.columns:
        data['SMA_20'] = ta.sma(data['Close'], length=20)
        data['SMA_50'] = ta.sma(data['Close'], length=50)
        data['SMA_200'] = ta.sma(data['Close'], length=200)
        sma_available = data[['SMA_20', 'SMA_50', 'SMA_200']].notna().any().any()

    # --- Calculate RSI ---
    if 'Close' in data.columns:
        data['RSI'] = ta.rsi(data['Close'], length=14)
        rsi_available = 'RSI' in data.columns and data['RSI'].notna().any()

    # --- Calculate MACD ---
    macd = ta.macd(data['Close'])
    if macd is not None and all(col in macd.columns for col in ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']):
        data['MACD'] = macd['MACD_12_26_9']
        data['MACD_signal'] = macd['MACDs_12_26_9']
        data['MACD_hist'] = macd['MACDh_12_26_9']
        macd_available = True

    # --- Calculate OBV ---
    if 'Close' in data.columns and 'Volume' in data.columns:
        data['OBV'] = ta.obv(data['Close'], data['Volume'])
        obv_available = 'OBV' in data.columns and data['OBV'].notna().any()

    # --- Calculate ADX ---
    adx = ta.adx(data['High'], data['Low'], data['Close'])
    if adx is not None and 'ADX_14' in adx.columns:
        data['ADX'] = adx['ADX_14']
        adx_available = True

    # --- Calculate Bollinger Bands ---
    bbands = ta.bbands(data['Close'], length=20, std=2)
    if bbands is not None and all(col in bbands.columns for col in ['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0']):
        data['upper_band'] = bbands['BBU_20_2.0']
        data['middle_band'] = bbands['BBM_20_2.0']
        data['lower_band'] = bbands['BBL_20_2.0']
        bbands_available = True

    # --- Resample data weekly ---
    data = data.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'SMA_20': 'last',
        'SMA_50': 'last',
        'SMA_200': 'last',
        'RSI': 'last',
        'MACD': 'last',
        'MACD_signal': 'last',
        'MACD_hist': 'last',
        'OBV': 'last',
        'ADX': 'last',
        'upper_band': 'last',
        'middle_band': 'last',
        'lower_band': 'last'
    })

    # --- Prepare data for analysis ---
    recent_data = data

    # --- Run your original analysis functions (these return text) ---
    results = {
        "bd_result": bollingerbands(ticker, recent_data[["Open", "High", "Low", "Close", "Volume", "upper_band", "middle_band", "lower_band"]].to_markdown()),
        "sma_result": SMA(ticker, recent_data[["Open", "High", "Low", "Close", "SMA_20", "SMA_50", "SMA_200"]].to_markdown()) if sma_available else "SMA analysis not available.",
        "rsi_result": RSI(ticker, recent_data[["Open", "High", "Low", "Close", "RSI"]].to_markdown()) if rsi_available else "RSI analysis not available.",
        "macd_result": MACD(ticker, recent_data[["Open", "High", "Low", "Close", "MACD", "MACD_signal", "MACD_hist"]].to_markdown()) if macd_available else "MACD analysis not available.",
        "obv_result": OBV(ticker, recent_data[["Open", "High", "Low", "Close", "Volume", "OBV"]].to_markdown()) if obv_available else "OBV analysis not available.",
        "adx_result": ADX(ticker, recent_data[["Open", "High", "Low", "Close", "ADX"]].to_markdown()) if adx_available else "ADX analysis not available."
    }

    availability = {
        "sma_available": sma_available,
        "rsi_available": rsi_available,
        "macd_available": macd_available,
        "obv_available": obv_available,
        "adx_available": adx_available,
        "bbands_available": bbands_available
    }

    indicator_scores = {k: [] for k in weights}

    for _, week in data.iterrows():
    # SMA
        if availability['sma_available'] and pd.notna(week['Close']) and pd.notna(week['SMA_20']):
            score = 1 if week['Close'] > week['SMA_20'] else -1
            indicator_scores['sma'].append(score)
        # RSI
        if availability['rsi_available'] and pd.notna(week['RSI']):
            if week['RSI'] > 55:
                score = 1
            elif week['RSI'] < 45:
                score = -1
            else:
                score = 0
            indicator_scores['rsi'].append(score)
        # MACD
        if availability['macd_available'] and pd.notna(week['MACD']) and pd.notna(week['MACD_signal']):
            score = 1 if week['MACD'] > week['MACD_signal'] else -1
            indicator_scores['macd'].append(score)
        # OBV
        if availability['obv_available'] and pd.notna(week['OBV']):
            if week['OBV'] > 0:
                score = 1
            elif week['OBV'] < 0:
                score = -1
            else:
                score = 0
            indicator_scores['obv'].append(score)
        # ADX
        if availability['adx_available'] and pd.notna(week['ADX']):
            score = 1 if week['ADX'] > 20 else -1
            indicator_scores['adx'].append(score)
        # BBands
        if availability['bbands_available'] and pd.notna(week['Close']) and pd.notna(week['middle_band']):
            score = 1 if week['Close'] > week['middle_band'] else -1
            indicator_scores['bbands'].append(score)

    # Aggregate: take the mean (average) score for each indicator
    import numpy as np
    final_scores = {}
    for k in indicator_scores:
        if indicator_scores[k]:  # If there are scores for that indicator
            final_scores[k] = np.mean(indicator_scores[k])
        else:
            final_scores[k] = 0

    # Now calculate weighted score as before, but using averages over weeks
    total_weight = sum(weights[k] for k in final_scores if availability.get(f"{k}_available", False))
    weighted_score = (
        sum(final_scores[k] * weights[k] for k in final_scores if availability.get(f"{k}_available", False)) / total_weight
        if total_weight > 0 else 0
    )

    print("Final Indicator Averages:", final_scores)
    print("Weighted Score:", weighted_score)
    

    # --- RETURN everything ---
    return results, recent_data, availability, weighted_score


# --- AI Comparison ---


def clean_html_response(response):
    # Remove markdown formatting from response
    if response.startswith("```html"):
        response = response.lstrip("```html").strip()
    if response.endswith("```"):
        response = response.rstrip("```").strip()
    return response

def fix_html_with_embedded_markdown(text):
    """
    Detects markdown sections embedded within mostly-HTML output,
    converts them to HTML, and replaces them in the text.
    """
    if not text:
        return text

    # Don't touch it if it's a fully valid HTML document
    if bool(re.search(r'<html', text, re.IGNORECASE)):
        return text

    # Pattern to detect markdown-style headings, lists, bold, etc.
    markdown_blocks = list(re.finditer(
        r'(?:(^|\n)(\s*)(#{1,6} .+|[-*+] .+|\d+\..+|>\s.+|\*\*.+\*\*|__.+__)([\s\S]+?))(?=\n{2,}|\Z)', 
        text,
        flags=re.MULTILINE
    ))

    # Convert and replace each markdown block
    for match in reversed(markdown_blocks):  # reversed to not break indices when replacing
        md_block = match.group(0).strip()
        # Only convert if not inside an HTML tag already
        if not re.match(r'<[a-z][^>]*>', md_block):
            html_block = markdown2.markdown(md_block)
            # Optionally strip <p> if markdown2 wraps the entire block
            if html_block.startswith('<p>') and html_block.endswith('</p>\n'):
                html_block = html_block[3:-5]
            # Replace markdown block with HTML
            start, end = match.span(0)
            text = text[:start] + html_block + text[end:]

    return text

def generate_news_html(gathered_data):
    # Use the system prompt and gathered data to generate the news HTML
    news_system_prompt = system_prompt_html

    user_message = f"The data to analyse: {json.dumps(gathered_data)}"
    
    # Call Claude API to generate the HTML with progress indicator
    with st.spinner("Generating investment analysis..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",  # Use the appropriate Claude model
                messages=[
                    {"role": "system", "content": news_system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            
            # Extract the response content
            html_content = response.choices[0].message.content
            return html_content
            
        except Exception as e:
            st.error(f"Error generating analysis: {e}")
            return None

# --- Streamlit App ---
def main():
    st.title("📊 Company Technical Indicator Comparison (Marketstack)")

    st.sidebar.subheader("Analysis Options")
    technical_analysis = st.sidebar.checkbox("Technical Analysis", help="Select to run technical analysis indicators")
    news_analysis = st.sidebar.checkbox("News Analysis", help="Select to run news sentiment analysis")

    tickers_input = st.sidebar.text_area("Enter tickers separated by commas", value="AAPL,MSFT,GOOGL")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    companies = {}
    if news_analysis:
        st.sidebar.subheader("Company Names for News Analysis")
        for ticker in tickers:
            companies[ticker] = st.sidebar.text_input(
                f"Company name for {ticker}", value=""
            )
    if technical_analysis:
        timeframe = st.sidebar.radio("Choose timeframe:", ("3 Months","6 Months","1 Year"), index=2)
        weight_choice = st.sidebar.radio("Choose weight scheme:", ("Default","Short Term","Long Term"), index=0)


    if st.sidebar.button("Run Analysis"):
        progress_bar = st.progress(0)
        
        all_gathered_data = []  # store results for all tickers
        if technical_analysis:

            for ticker in tickers:
                company = ticker  # extend with name lookup if you want
                data = fetch_marketstack_data(ticker, timeframe)
                if data is None or data.empty:
                    st.error(f"No data found for {ticker}")
                    continue

                with st.expander(f"Downloading Data for {ticker}..."):
                    update_progress(progress_bar, 50, 50, f"Analyzing {ticker}...")
                    results, recent_data, availability, weighted_score = calculate_technical_indicators(
                        data, ticker, weight_choice=weight_choice
                    )

                bd_result = results["bd_result"]
                sma_result = results["sma_result"]
                rsi_result = results["rsi_result"]
                macd_result = results["macd_result"]
                obv_result = results["obv_result"]
                adx_result = results["adx_result"]

            gathered_data = {
                "Ticker": ticker,
                "Company": company,
                "Timeframe": timeframe,
                "data": recent_data.to_dict(orient="records"),
                "Position_type": weight_choice,
                "weighted_score": weighted_score,
                "Results": {
                    "SMA Results": sma_result,
                    "RSI Results": rsi_result,
                    "MACD Results": macd_result,
                    "OBV Results": obv_result,
                    "BD Results": bd_result,
                    "ADX Results": adx_result
                }
            }

            all_gathered_data.append(gathered_data)

        # ---- Now generate a combined summary once ----
            if all_gathered_data:
                summary = SUMMARY2(all_gathered_data)   # pass the list instead of a single dict
                html_output_no_fix = clean_html_response(summary)
                html_output = fix_html_with_embedded_markdown(html_output_no_fix)

                update_progress(progress_bar, 100, 100, "Finished All Tickers...")

                # Render in app
                st.components.v1.html(html_output, height=700, width=1400, scrolling=True)

                # Extract plain text version
                soup = BeautifulSoup(html_output, "html.parser")
                plain_text = soup.get_text(separator='\n')

                # Save in session state
                st.session_state["gathered_data"] = all_gathered_data
                st.session_state["analysis_complete"] = True
                st.session_state["html_output"] = html_output
                st.session_state["plain_text"] = plain_text

                st.success("Stock analysis completed for all tickers! You can now proceed to the AI Chatbot.")

                # Download buttons
                st.download_button("Download as HTML", st.session_state["html_output"], "stock_analysis_summary.html", "text/html")
                st.download_button("Download as Plain Text", st.session_state["plain_text"], "stock_analysis_summary.txt", "text/plain")

                # Reset button
                if st.button("Run Another Analysis"):
                    st.session_state.technical_analysis = False
                    st.session_state.news_and_events = False
                    st.session_state["run_analysis_complete"] = False
                    st.experimental_rerun()
        if news_analysis:
            timeframe_map = {
                    "3 Months": "3m",
                    "6 Months": "6m",
                    "1 Year": "1y"
                }

            timeframe_key = timeframe_map.get(timeframe)
            gathered_data_list = []

            for ticker, company in zip(tickers, companies):  # tickers and companies should align
                with st.expander(f"Downloading Data for {ticker}"):
                    update_progress(progress_bar, 30, 30, f"Gathering News Data for {ticker}...")

                    news_data = get_news_sentiment_gathered_data(
                        ticker=ticker,
                        period=timeframe_key,
                        company_name=company,
                        alpha_vantage_api_key=st.secrets["ALPHA_VANTAGE_API_KEY"],
                        openai_api_key=st.secrets["OPENAI_API_KEY"],
                    )

                    print("News Data:", news_data)  # Debug print to check news data

                    update_progress(progress_bar, 50, 50, f"Analysing News Data for {ticker}...")

                    # Append news_data (already in correct schema) to master list
                    if news_data:
                        gathered_data_list.append(news_data)
                    else:
                        st.warning(f"No news data found for {ticker}")

            news_html = generate_news_html(gathered_data_list)
            html_output_no_fix = clean_html_response(news_html)
            html_output = fix_html_with_embedded_markdown(html_output_no_fix)
            update_progress(progress_bar, 100, 100, "")
            st.components.v1.html(html_output, height=700, width=1400, scrolling=True)
            
            # Placeholder for future news analysis implementation

if __name__ == "__main__":
    main()
