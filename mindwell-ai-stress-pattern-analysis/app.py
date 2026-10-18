from pathlib import Path
import pandas as pd, streamlit as st, plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
st.set_page_config(page_title="MindWell AI",layout="wide")
st.title("🌿 MindWell AI: Well-Being & Stress Pattern Analysis")
st.caption("Synthetic educational data only. Not a mental-health diagnostic tool.")
p=Path("data/wellbeing.csv")
if not p.exists(): st.warning("Run `python generate_data.py` first."); st.stop()
df=pd.read_csv(p); target="self_reported_stress_category"
a,b,c=st.columns(3); a.metric("Synthetic records",len(df)); b.metric("Average sleep",f"{df.sleep_hours.mean():.1f} hrs"); c.metric("Average activity",f"{df.activity_minutes.mean():.0f} min")
l,r=st.columns(2)
l.plotly_chart(px.bar(df[target].value_counts().rename_axis("category").reset_index(name="records"),x="category",y="records"),use_container_width=True)
r.plotly_chart(px.scatter(df,x="sleep_hours",y="workload_rating",color=target),use_container_width=True)
num=["sleep_hours","activity_minutes","workload_rating","screen_hours"]; cat=["routine_type"]; features=num+cat
prep=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
model=Pipeline([("prep",prep),("model",RandomForestClassifier(n_estimators=150,random_state=42,class_weight="balanced"))])
Xtr,Xte,ytr,yte=train_test_split(df[features],df[target],test_size=.25,random_state=42,stratify=df[target])
model.fit(Xtr,ytr); pred=model.predict(Xte)
st.metric("Holdout accuracy on synthetic labels",f"{accuracy_score(yte,pred):.1%}")
st.dataframe(pd.DataFrame(classification_report(yte,pred,output_dict=True,zero_division=0)).transpose().round(3),use_container_width=True)
st.dataframe(df,use_container_width=True,hide_index=True)
st.download_button("Download synthetic data CSV",df.to_csv(index=False).encode(),"mindwell_wellbeing.csv","text/csv")
