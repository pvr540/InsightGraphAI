# LangGraph + Ad Manager Order ID Bundle

## What this bundle includes
- LangGraph workflow with graph flow + trace
- MCP server tools
- Google Ad Manager order-id report tool
- JSON output for order metrics
- SQL + utility tools + Auto Compare mode

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python3 db_setup.py
   python schema_indexer.py
   ```

2. Create a `googleads.yaml` file for Ad Manager auth.
   Google Ad Manager SOAP API uses OAuth2, and you must enable API access and add a service account user in your Ad Manager network. The reporting API supports report jobs, status polling, and downloading completed reports. Reports are gzip-compressed by default unless you change the download options. See Google’s Ad Manager auth and reporting docs. citeturn553703search13turn553703search3turn553703search1

3. Set environment variables:
   ```bash
   export ADM_NETWORK_CODE="your_network_code"
   export ADM_START_DATE="2024-01-01"
   export ADM_END_DATE="2024-01-31"
   export ADM_API_VERSION="v202602"
   ```

4. Run the app:
   ```bash
   streamlit run langgraph_streamlit_app.py
   ```

## Demo prompt for order ID
- Show metrics for order 123456