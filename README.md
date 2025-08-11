# SEC Filings QA Agent

This project is a question-answering system that analyzes SEC filings to answer complex financial research questions.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd SEC_Filings_QA_Agent
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Download SEC filings:**
    *   Run the `fetch_data.py` script to download the data. You can modify the script to change the companies or filing types.
    ```bash
    python fetch_data.py
    ```

4.  **Create the Vector Store:**
    *   Run the `create_vector_store.py` script to build the vector store from the downloaded data.
    ```bash
    python create_vector_store.py
    ```

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## Supported SEC Filing Types

The agent is configured to process the following types of SEC filings:

*   **Registration Statements:** S-1, S-4
*   **Periodic Reports:** 10-K, 10-Q, 8-K
*   **Insider Reporting:** 3, 4, 5
*   **Beneficial Ownership Reports:** 13D
*   **Proxy Statements:** DEF 14A
*   **Other Filings:** 144, 11-K, 13F
*   **Foreign Private Issuer Filings:** 1-F, 20-F

## Supported Companies

The agent is configured to download filings for the following companies:

| Company Name | Ticker |
| :--- | :--- |
| Apple Inc. | AAPL |
| Microsoft Corporation | MSFT |
| Alphabet Inc. | GOOGL |
| Amazon.com, Inc. | AMZN |
| Meta Platforms, Inc. | META |
| Tesla, Inc. | TSLA |
| NVIDIA Corporation | NVDA |
| JPMorgan Chase & Co. | JPM |
| Johnson & Johnson | JNJ |
| Walmart Inc. | WMT |
| The Procter & Gamble Company | PG |
| UnitedHealth Group Incorporated | UNH |
| The Home Depot, Inc. | HD |
| Visa Inc. | V |
| Mastercard Incorporated | MA |
| Bank of America Corporation | BAC |
| Exxon Mobil Corporation | XOM |
| Chevron Corporation | CVX |
| The Coca-Cola Company | KO |
| PepsiCo, Inc. | PEP |
| McDonald's Corporation | MCD |
| The Walt Disney Company | DIS |
| Netflix, Inc. | NFLX |
| Adobe Inc. | ADBE |
| Salesforce, Inc. | CRM |
| Oracle Corporation | ORCL |
| Cisco Systems, Inc. | CSCO |
| Intel Corporation | INTC |
| Pfizer Inc. | PFE |
| Merck & Co., Inc. | MRK |
