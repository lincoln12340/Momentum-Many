import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from openai import OpenAI
import plotly.express as px

api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key= api_key)

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
        try:
            if timeframe == "3 Months":
                data = yf.download(ticker, period="3mo")
            elif timeframe == "6 Months":
                data = yf.download(ticker, period="6mo")
            elif timeframe == "1 Year":
                data = yf.download(ticker, period="1y")

            data.columns = data.columns.droplevel(1)
            if not data.empty:
                validity_results.append({"Ticker": ticker, "Valid": True})
                data_dict[ticker] = data
            else:
                validity_results.append({"Ticker": ticker, "Valid": False})
        except Exception:
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
        model="gpt-4o",  # Ensure that you use a model available in your OpenAI subscription
        messages=[
            {
                "role": "system",
                "content": (
                    """Purpose of the Analysis 
                    The output is designed to:

                    Provide actionable insights for investment analysts.
                    Focus on momentum-based performance metrics, such as weighted scores.
                    Offer structured, professional-level analysis, combining individual company, group-level, and sector-specific evaluations.
                    2. Audience and Tone
                    Audience: Investment analysts or finance professionals seeking insights for decision-making.
                    Tone: Professional, data-driven, and insightful. Use precise language that is easy to follow while maintaining analytical depth.
                    3. Output Structure
                    Introduction

                    Briefly describe the dataset and the purpose of the analysis.
                    Explain the metrics used (e.g., weighted scores, Rate of Change (ROC), Momentum, RSI) and their relevance.
                    Highlight the key areas of focus: individual company performance, group-level trends, sector-specific insights, and cross-group comparisons.
                    Example:
                    “This analysis evaluates two groups of companies based on their weighted scores, derived from key financial indicators. The objective is to identify momentum leaders, sectoral trends, and actionable investment opportunities.”

                    Group-Level Momentum Comparison

                    Compare statistical metrics for each group:
                    Mean, range, and standard deviation of weighted scores.
                    Use percentage differences to emphasize comparative insights.
                    Summarize which group demonstrates stronger momentum and explain why.
                    Example:

                    “Group 2 outperforms Group 1 by 25.43%, driven by higher variability and standout performers within the pharmaceuticals sector.”
                    Individual Company Analysis

                    List each company, grouped by dataset, and include:
                    Ticker: Stock symbol for identification.
                    Company Name: Translate tickers to company names if possible.
                    Sector: Industry classification.
                    Weighted Score: Performance metric.
                    Insight: Brief explanation of its momentum performance and potential drivers.
                    Example:

                    CRSP (CRISPR Therapeutics) – Biotechnology
                    Weighted Score: 0.408
                    Momentum Insight: “CRISPR leads Group 1, driven by advancements in gene-editing technology, reflecting strong sectoral innovation.”
                    Sector-Specific Analysis

                    Evaluate sector representation within each group.
                    Discuss trends impacting sector momentum (e.g., innovation, market stability, macroeconomic factors).
                    Highlight diversification or concentration effects on overall group performance.
                    Example:

                    “Group 1 is biotechnology-heavy, resulting in consistent but modest momentum. In contrast, Group 2 includes pharmaceuticals, biotechnology, and healthcare services, offering higher momentum and diversification.”
                    Cross-Group Sector Comparison

                    Compare sectoral dynamics across groups:
                    Which sectors drive momentum in each group?
                    What are the strengths and weaknesses of each group’s sector composition?
                    Discuss trends and differences in sector contributions to performance.
                    Example:

                    “Pharmaceutical giants in Group 2 outpace Group 1’s biotech companies, reflecting stability from established revenue streams.”
                    Key Takeaways and Recommendations

                    Summarize the key findings, focusing on:
                    Group-level momentum.
                    Sector trends.
                    Top-performing companies.
                    Provide actionable investment recommendations:
                    Which group, sector, or company should be prioritized for different strategies (e.g., high-growth, risk-averse)?
                    Example:

                    “Group 2 is the clear leader in momentum, with Regeneron and Vertex offering the strongest growth opportunities. Analysts should prioritize Group 2 while maintaining exposure to Group 1 for stability.”
                    4. Instructions for Analysis Workflow
                    Load and Preprocess Data:

                    Import datasets and ensure columns are standardized (e.g., ticker, weighted score, sector).
                    Compute statistical metrics (mean, range, standard deviation) for each group.
                    Calculate Comparisons:

                    Use percentage differences to compare group averages and identify standout performers.
                    Company-Specific Analysis:

                    Match tickers to company names and sectors if not directly provided.
                    Generate concise momentum insights for each company.
                    Sector-Specific Analysis:

                    Categorize companies by sector and evaluate sector-wide trends influencing performance.
                    Cross-Group Analysis:

                    Compare sectoral representation and contributions to overall momentum.
                    Generate Output:

                    Structure the output according to the template provided, ensuring logical flow and clear insights.
                    5. Notes for Adaptability
                    Dynamic Structure: Ensure the structure adapts to the dataset content (e.g., if new sectors appear, integrate them into sector-specific and cross-group analysis).
                    Language Precision: Always aim for clarity, using terminology familiar to investment analysts without unnecessary complexity.
                    Actionable Insights: Emphasize recommendations that directly support investment decision-making.
                    """
                                        #Add Press releases, investor oppinions (X), First World Pharma, Bloomberg, Market Watch, seperate segment,add sources, add graphs
                    
                ),
            },
            {
                "role": "user",
                "content": (
                    f""""Using the two datasets provided below, perform a comprehensive momentum analysis. Structure the output as follows:

                    Introduction: Briefly describe the datasets and analysis objectives.
                    Group-Level Momentum Comparison: Compare the performance of the two groups, highlighting key statistics (e.g., mean, range, standard deviation) and percentage differences. Identify which group demonstrates stronger momentum.
                    Individual Company Analysis: Analyze each company within the groups. Include the ticker, company name, sector, weighted score, and a brief explanation of the company’s momentum.
                    Sector-Specific Analysis: Evaluate trends within each group by sector. Discuss how sector representation influences momentum.
                    Cross-Group Sector Comparison: Compare sector dynamics across the two groups. Highlight strengths, weaknesses, and trends contributing to performance differences.
                    Key Takeaways and Recommendations: Summarize findings, identify top performers, and provide actionable investment recommendations.
                    Tone: Use professional, analytical language targeted at investment analysts. Focus on delivering clear, data-driven insights and actionable recommendations

                    Group 1 Data: {portfolio_results}
                    Group 2 Data: {benchmark_results}
                    """
                ),
            },
        ]
    )

    # Extract and return the AI-generated response
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
        benchmark_file = st.sidebar.file_uploader("Upload Group 2 CSV", type="csv")
        if portfolio_file and benchmark_file:
            portfolio_df = pd.read_csv(portfolio_file)
            benchmark_df = pd.read_csv(benchmark_file)
            if 'Ticker' not in portfolio_df.columns or 'Ticker' not in benchmark_df.columns:
                st.error("Both CSVs must contain a 'Ticker' column.")
                return
    else:
        portfolio_tickers = st.sidebar.text_area("Enter Portfolio Tickers (comma-separated)")
        benchmark_tickers = st.sidebar.text_area("Enter Benchmark Tickers (comma-separated)")
        portfolio_df = pd.DataFrame({"Ticker": portfolio_tickers.split(",")})
        benchmark_df = pd.DataFrame({"Ticker": benchmark_tickers.split(",")})

    if st.sidebar.button("Run Analysis"):
        indicator_weights = {"RSI": 0.1, "ROC": 0.8, "Momentum": 0.1}
        progress_bar = st.progress(0)
        with st.expander("Progress Tracker"):
            st.write("Starting Analysis...")
            with st.spinner("Processing Portfolio Data..."):
                portfolio_results = calculate_scores(portfolio_df,indicator_weights,timeframe)
                progress_bar.progress(50)

            st.write("Finished Portfolio Analysis...")

            with st.spinner("Analyzing Benchmark Data..."):
                benchmark_results = calculate_scores(benchmark_df,indicator_weights,timeframe)
                progress_bar.progress(75)

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
