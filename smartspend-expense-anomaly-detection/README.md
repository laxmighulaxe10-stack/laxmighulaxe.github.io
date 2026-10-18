# SmartSpend: AI-Powered Expense Anomaly Detection

Portfolio demo using Python, Pandas, Scikit-learn, Isolation Forest and Streamlit. Sample transactions are synthetic, not real financial data.

## Run
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python generate_data.py
streamlit run app.py
```
## What it demonstrates
Data cleaning, feature engineering, unsupervised anomaly detection, visual review and CSV export. An anomaly flag is a review signal, not proof of fraud.
