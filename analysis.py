import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from openai import OpenAI
import plotly.express as px
from alpha_vantage.timeseries import TimeSeries
import time

api_key = st.secrets["OPENAI_API_KEY"]
alpha_vantage_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
client = OpenAI(api_key= api_key)

@st.cache_data(ttl=3600)
def fetch_alpha_vantage_data(ticker, period):
    """Fetch data from Alpha Vantage and filter by period"""
    ts = TimeSeries(key=alpha_vantage_key, output_format='pandas')
    
    try:
        # Get full daily data (we'll filter it later)
        data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
        data.index = pd.to_datetime(data.index)
        
        # Filter based on selected period
        today = pd.Timestamp.today()
        period_map = {
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365
        }
        cutoff_days = period_map.get(period, 365)
        cutoff_date = today - pd.Timedelta(days=cutoff_days)

        filtered_data = data[data.index >= cutoff_date]

        
        #filtered_data = data.last(period_map.get(period, "1Y"))
        
        # Rename columns to match yfinance format
        filtered_data = filtered_data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        st.write(filtered_data)
        
        return filtered_data.sort_index()
    
    except Exception as e:
        st.error(f"Alpha Vantage Error: {str(e)}")
        return None

def gather_data(portfolio_results, benchmark_results, Results):
    gathered_data = {
        
        "Portfolio Results": portfolio_results.to_dict(orient="records"),
        "Benchmark Results": benchmark_results.to_dict(orient="records"),
        "Overall Results": Results
    }

    return gathered_data
# Function to calculate indicators
def calculate_indicators(data):
    data['RSI'] = ta.rsi(data['Close'], length=14)
    data['ROC'] = ta.roc(data['Close'], length=12)
    data['Momentum'] = ta.mom(data['Close'], length=10)
    return data

# Function to calculate indicators with weights
def calculate_indicators_with_weights(data, weights):
    data = calculate_indicators(data)
    data['RSI_norm'] = (data['RSI'] - data['RSI'].min()) / (data['RSI'].max() - data['RSI'].min())
    data['ROC_norm'] = (data['ROC'] - data['ROC'].min()) / (data['ROC'].max() - data['ROC'].min())
    data['Momentum_norm'] = (data['Momentum'] - data['Momentum'].min()) / (data['Momentum'].max() - data['Momentum'].min())
    data['Weighted_Score'] = (
        weights['RSI'] * data['RSI_norm'] +
        weights['ROC'] * data['ROC_norm'] +
        weights['Momentum'] * data['Momentum_norm']
    )
    return data[['Close', 'RSI', 'ROC', 'Momentum', 'Weighted_Score']]

# Function to check ticker validity and download data
def check_ticker_validity_and_download(tickers,timeframe):
    validity_results = []
    data_dict = {}
    
    for ticker in tickers:
        
            #if timeframe == "3 Months":
                #data = yf.download(ticker, period="3mo")
            #elif timeframe == "6 Months":
                #data = yf.download(ticker, period="6mo")
            #elif timeframe == "1 Year":
                #data = yf.download(ticker, period="1y")

            #data.columns = data.columns.droplevel(1)
        data = fetch_alpha_vantage_data(ticker, timeframe)
           
            
        if not data.empty:
            validity_results.append({"Ticker": ticker, "Valid": True})
            data_dict[ticker] = data
        else:
            validity_results.append({"Ticker": ticker, "Valid": False})


    return pd.DataFrame(validity_results), data_dict

# Function to calculate scores
def calculate_scores(components, indicator_weights,timeframe):
    valid_tickers_df, data_dict = check_ticker_validity_and_download(components['Ticker'],timeframe)
    merged_data = pd.merge(components, valid_tickers_df, on="Ticker")
    valid_tickers_data = merged_data[merged_data['Valid'] == True]
    results = []
    for _, row in valid_tickers_data.iterrows():
        ticker = row['Ticker']
        try:
            data = data_dict[ticker]
            data_weighted = calculate_indicators_with_weights(data, indicator_weights)
            avg_weighted_score = data_weighted['Weighted_Score'].mean()
            results.append({"Ticker": ticker, "Average_Weighted_Score": avg_weighted_score})
        except Exception:
            results.append({"Ticker": ticker, "Average_Weighted_Score": None})
    return pd.DataFrame(results)

# Function to compare portfolio and benchmark
def portfolio_vs_benchmark(portfolio_results, benchmark_results):
    
    #portfolio_score = portfolio_results['Average_Weighted_Score'].mean()
    #benchmark_score = benchmark_results['Average_Weighted_Score'].mean()

    # Determine momentum direction
    #portfolio_direction = "upward" if portfolio_score > benchmark_score else "downward"

    # Identify single components impacting results significantly
    #top_portfolio_contributors = portfolio_results.nlargest(3, 'Average_Weighted_Score')
    #top_benchmark_contributors = benchmark_results.nlargest(3, 'Average_Weighted_Score')
    

  

    chat_completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": (
                """Purpose of the Analysis 
                The output is designed to:

                Provide actionable insights for investment analysts.
                Focus on momentum-based performance metrics, such as weighted scores.
                Offer structured, professional-level analysis across individual tickers and sector insights within a single dataset.

                Audience and Tone:
                Audience: Investment analysts or finance professionals.
                Tone: Professional, data-driven, and insightful.

                Output Structure:
                1. Introduction
                - Describe the dataset and purpose of the analysis.
                - Explain the metrics used: Weighted Score (derived from RSI, Rate of Change, Momentum).

                2. Ticker Performance Comparison
                - Compare all tickers statistically.
                - Report the mean, range, and standard deviation of scores.
                - Rank tickers by momentum performance.
                - Identify top and bottom performers, and any clustering of strong/weak momentum.

                3. Individual Ticker Analysis
                - For each ticker, present:
                  • Ticker Symbol
                  • Company Name (if known)
                  • Sector (if available)
                  • Weighted Score
                  • One-sentence performance insight

                4. Sector Analysis
                - Group tickers by sector (if sector info is available).
                - Identify sector-wide performance trends.
                - Note any standout or underperforming sectors.

                5. Key Takeaways & Investment Recommendations
                - Summarize strongest-performing tickers.
                - Highlight sector trends.
                - Offer actionable ideas (e.g., tickers to monitor, momentum leaders).

                Notes:
                - Adjust section structure if some metadata (e.g., sectors) is missing.
                - Avoid comparing to a benchmark—focus is on intra-group dynamics and ranking.
                - Maintain a concise, clear tone suitable for analyst reporting.
                """
            ),
        },
        {
            "role": "user",
            "content": (
                f"""Please analyze the following ticker dataset using momentum-based metrics and provide structured insights.

                Dataset: {portfolio_results}

                Goals:
                - Compare tickers against one another based on average weighted score.
                - Rank the tickers and identify leaders/laggards.
                - Provide sector-level analysis if sector data is available.
                - Offer investment recommendations based on the momentum trends observed.
                """
            ),
        },
    ]
)

response = chat_completion.choices[0].message.content
return response


# Streamlit App
def main():

    if "run_analysis_complete" not in st.session_state:
        st.session_state["run_analysis_complete"] = False

    st.title("Group Comparison Analysis")

    st.sidebar.subheader("Select Timeframe for Analysis")
    timeframe = st.sidebar.radio(
        "Choose timeframe:",
        ( "3 Months", "6 Months", "1 Year"),
        index=2,
        help="Select the period of historical data for the stock analysis")
        
    st.sidebar.header("Input Options")
    input_method = st.sidebar.radio("How would you like to input the data?", ("Upload CSVs", "Enter Manually"))

    if input_method == "Upload CSVs":
        portfolio_file = st.sidebar.file_uploader("Upload Group 1 Tickers CSV", type="csv")
        #benchmark_file = st.sidebar.file_uploader("Upload Group 2 Tickers CSV", type="csv")
        if portfolio_file:
            portfolio_df = pd.read_csv(portfolio_file)
            #benchmark_df = pd.read_csv(benchmark_file)
            if 'Ticker' not in portfolio_df.columns:
                st.error("CSV must contain a 'Ticker' column.")
                return
    else:
        portfolio_tickers = st.sidebar.text_area("Enter Portfolio Tickers (comma-separated)")
        #benchmark_tickers = st.sidebar.text_area("Enter Benchmark Tickers (comma-separated)")
        portfolio_df = pd.DataFrame({"Ticker": portfolio_tickers.split(",")})
        #benchmark_df = pd.DataFrame({"Ticker": benchmark_tickers.split(",")})

    if st.sidebar.button("Run Analysis"):
        indicator_weights = {"RSI": 0.1, "ROC": 0.8, "Momentum": 0.1}
        progress_bar = st.progress(0)
        with st.expander("Progress Tracker"):
            st.write("Starting Analysis...")
            with st.spinner("Processing Portfolio Data..."):
                st.write(portfolio_df)
                portfolio_results = calculate_scores(portfolio_df,indicator_weights,timeframe)
                progress_bar.progress(50)

            st.write("Finished Portfolio Analysis...")

            #with st.spinner("Analyzing Benchmark Data..."):
                #benchmark_results = calculate_scores(benchmark_df,indicator_weights,timeframe)
                #progress_bar.progress(75)

            st.write("Comparing with Benchmark...")

            with st.spinner("Generating Final Report..."):
                analysis = portfolio_vs_benchmark(portfolio_results, benchmark_results)
                progress_bar.progress(100)

            st.write("Completed")

        

        with st.expander("Analysis Results"):
            #st.markdown("### Portfolio Overall Score: {:.2f}".format(portfolio_score))
            #st.markdown("### Benchmark Overall Score: {:.2f}".format(benchmark_score))
            st.write(analysis)

        with st.expander("Graphs"):
            st.write(portfolio_results)

            st.subheader("Portfolio Contribution by Ticker")
            fig = px.bar(
                portfolio_results,
                x='Ticker',
                y='Average_Weighted_Score',
                title='Portfolio Contribution by Ticker',
                labels={'Average_Weighted_Score': 'Average Weighted Score'}
            )
            st.plotly_chart(fig)

            st.subheader("Benchmark Contribution by Ticker")
            fig = px.bar(
                benchmark_results,
                x='Ticker',
                y='Average_Weighted_Score',
                title='Benchmark Contribution by Ticker',
                labels={'Average_Weighted_Score': 'Average Weighted Score'}
            )
            st.plotly_chart(fig)

        ovr_result = gather_data(portfolio_results, benchmark_results, analysis)
        st.session_state["gathered_data"] = ovr_result
        st.session_state["analysis_complete"] = True  # Mark analysis as complete
        st.success("Group analysis completed! You can now proceed to the AI Chatbot.")

if __name__ == "__main__":
    main()
