from pathlib import Path
import numpy as np, pandas as pd, streamlit as st, plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
st.set_page_config(page_title="SmartSpend",layout="wide")
st.title("💸 SmartSpend: Expense Anomaly Detection")
st.caption("Synthetic learning demo. Flags need human review and are not proof of fraud.")
p=Path("data/transactions.csv")
if not p.exists(): st.warning("Run `python generate_data.py` first."); st.stop()
df=pd.read_csv(p); df["transaction_date"]=pd.to_datetime(df["transaction_date"])
df["log_amount"]=np.log1p(df["amount"]); df["day_of_week"]=df["transaction_date"].dt.dayofweek
cont=st.sidebar.slider("Expected anomaly fraction",.01,.10,.03,.01)
X=pd.get_dummies(df[["log_amount","transaction_hour","day_of_week","category","payment_method"]])
X=StandardScaler().fit_transform(X)
m=IsolationForest(contamination=cont,random_state=42).fit(X)
df["anomaly_score"]=-m.score_samples(X); df["status"]=np.where(m.predict(X)==-1,"Review","Typical")
a,b,c=st.columns(3); a.metric("Transactions",len(df)); b.metric("Flagged for review",int((df.status=="Review").sum())); c.metric("Median amount",f"₹{df.amount.median():,.0f}")
l,r=st.columns(2)
l.plotly_chart(px.histogram(df,x="amount",color="status",nbins=35),use_container_width=True)
r.plotly_chart(px.bar(df.groupby(["category","status"]).size().reset_index(name="count"),x="category",y="count",color="status",barmode="group"),use_container_width=True)
flag=df[df.status=="Review"].sort_values("anomaly_score",ascending=False)
st.dataframe(flag,use_container_width=True,hide_index=True)
st.download_button("Download flagged CSV",flag.to_csv(index=False).encode(),"flagged_transactions.csv","text/csv")
