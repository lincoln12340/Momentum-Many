from datetime import date
from openai import OpenAI
import streamlit as st
import json

OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_KEY)



def SUMMARY2(gathered_data):

    
    today = date.today()
    formatted = today.strftime('%Y-%m-%d')

    # --- New logic: support multiple companies ---
    if not isinstance(gathered_data, list):
        gathered_data = [gathered_data]

    system_prompt = f"""As an AI assistant for traders and investors, your task is to produce a structured comparative technical market analysis in valid HTML format.

    You will receive JSON containing **multiple companies**, each with:
    - Ticker, Company, Timeframe, weighted_score, and Results (indicator summaries).

    For each company:
    - Derive Momentum as:
    > Upward if weighted_score > 0.05  
    > Downward if weighted_score < -0.05  
    > Neutral otherwise  

    **Output Requirements:**
    1. At the very top, generate a **Momentum Ranking Table**:
    - Columns: Rank, Company, Ticker, Weighted Score, Momentum
    - Sort companies by weighted_score (highest to lowest).
    - Style the momentum column with the same CSS classes: `buy` (Upward), `hold` (Neutral), `sell` (Downward).

    2. Then, for each company, generate a full analysis block using <div class="container">:
    - Include ticker, company, timeframe, executive summary, momentum, and indicators.
    - Group all technical analysis by **weeks**, showing the analyzed date range.
    - Executive summary must explicitly note the weeks covered and that momentum is weekly (medium-term trends).
    - Indicators (SMA, RSI, MACD, OBV, ADX, Bollinger Bands) must use an <ol> with:
        * Current values
        * Position/trend vs price
        * Signal/crossover events
        * A bold summary conclusion

    **HTML Template (including Momentum Ranking Table and full CSS):**

    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Comparative Technical Momentum Analysis</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 0px;
                background-color: transparent;
            }}
            .container {{
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                padding: 30px;
                margin-bottom: 30px;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            h2 {{
                color: #2c3e50;
                border-left: 5px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
                background-color: #f8f9fa;
                padding: 10px 15px;
                border-radius: 0 5px 5px 0;
            }}
            h3 {{
                color: #2c3e50;
                margin-top: 20px;
                border-bottom: 1px dashed #ddd;
                padding-bottom: 5px;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }}
            .momentum {{
                font-weight: bold;
                font-size: 1.1em;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
                text-align: center;
            }}
            .buy {{
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }}
            .hold {{
                background-color: #fff3cd;
                color: #856404;
                border: 1px solid #ffeeba;
            }}
            .sell {{
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }}
            .summary-box {{
                background-color: #e8f4fd;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
                border-radius: 0 5px 5px 0;
            }}
            .timeframe {{
                font-weight: bold;
                color: #2c3e50;
                background-color: #e8f4fd;
                padding: 5px 10px;
                border-radius: 3px;
                display: inline-block;
                margin-bottom: 15px;
            }}
            .indicator {{
                margin-bottom: 20px;
                padding: 15px;
                border-radius: 5px;
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
            }}
            .indicator h4 {{
                margin-top: 0;
                color: #2980b9;
            }}
            .footnote {{
                font-size: 0.9em;
                font-style: italic;
                color: #6c757d;
                margin-top: 30px;
                padding-top: 15px;
                border-top: 1px solid #dee2e6;
            }}
            .highlight {{
                background-color: #ffeaa7;
                padding: 8px 12px;
                border-radius: 5px;
                display: block;
                margin: 15px 0 0 0;
                font-size: 1em;
            }}
            /* Responsive Design */
            @media (max-width: 768px) {{
                .container {{
                    padding: 10px;
                }}
                h1, h2 {{
                    font-size: 1.3em;
                    padding-left: 8px;
                    padding-right: 8px;
                }}
                .section {{
                    padding: 10px;
                }}
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 16px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f8f9fa;
                color: #2c3e50;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <h1>Comparative Technical Momentum Analysis</h1>

        <!-- Momentum Ranking Table -->
        <section class="section">
            <h2>Momentum Ranking</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Company</th>
                        <th>Ticker</th>
                        <th>Weighted Score</th>
                        <th>Momentum</th>
                    </tr>
                </thead>
                <tbody>
                    [MOMENTUM_RANKING_ROWS]
                </tbody>
            </table>
        </section>

        <!-- Individual company analysis blocks -->
        [COMPANY_ANALYSES]

        <div class="footnote">
            <p>This comparative momentum analysis was generated on {formatted}. Always consider multiple sources and your personal risk tolerance before investing.</p>
        </div>
    </body>
    </html>
    """

    user_message = f"The data to analyse: {json.dumps(gathered_data)}"
    chat_completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                # System message to define the assistant's behavior
            {
                    "role": "system",
                    "content":  system_prompt
                                    
            },
            # User message with a prompt requesting stock analysis for a specific company
            {
                "role": "user",
                "content": user_message
                    
            },
        ]
    )

# Output the AI's response
    response = chat_completion.choices[0].message.content

    return response
