from pathlib import Path
import numpy as np, pandas as pd
rng=np.random.default_rng(42); n=1200
cats=np.array(["Groceries","Transport","Dining","Shopping","Bills","Travel"])
category=rng.choice(cats,n,p=[.24,.18,.17,.18,.16,.07])
base={"Groceries":650,"Transport":180,"Dining":420,"Shopping":1100,"Bills":1500,"Travel":3200}
amount=np.array([rng.lognormal(np.log(base[c]),.45) for c in category])
dates=pd.Timestamp("2025-01-01")+pd.to_timedelta(rng.integers(0,365,n),unit="D")
hours=rng.integers(7,23,n); idx=rng.choice(n,24,replace=False); amount[idx]*=rng.uniform(4,10,len(idx)); hours[idx[:8]]=rng.choice([0,1,2,3,4],8)
df=pd.DataFrame({"transaction_id":[f"TXN{i:05d}" for i in range(1,n+1)],"transaction_date":dates.strftime("%Y-%m-%d"),"transaction_hour":hours,"category":category,"amount":amount.round(2),"payment_method":rng.choice(["UPI","Card","Net Banking","Cash"],n)})
Path("data").mkdir(exist_ok=True); df.to_csv("data/transactions.csv",index=False); print("Generated synthetic data/transactions.csv")
