from pathlib import Path
import numpy as np, pandas as pd
rng=np.random.default_rng(21); n=900
sleep=np.clip(rng.normal(7.1,1.2,n),3,10.5); activity=np.clip(rng.normal(45,22,n),0,140)
work=np.clip(rng.normal(6,2.1,n),1,10); screen=np.clip(rng.normal(5.5,2,n),.5,14)
routine=rng.choice(["Consistent","Mixed","Irregular"],n,p=[.35,.45,.20])
penalty=pd.Series(routine).map({"Consistent":-.5,"Mixed":0.,"Irregular":.8}).to_numpy()
score=5.5+(7-sleep)*.85+(work-5)*.45+(screen-5)*.18-(activity-40)*.012+penalty+rng.normal(0,1.1,n)
stress=np.where(score<5,"Lower",np.where(score<7.5,"Moderate","Higher"))
df=pd.DataFrame({"record_id":[f"MW{i:04d}" for i in range(1,n+1)],"sleep_hours":sleep.round(1),"activity_minutes":activity.round().astype(int),"workload_rating":work.round(1),"screen_hours":screen.round(1),"routine_type":routine,"self_reported_stress_category":stress})
Path("data").mkdir(exist_ok=True); df.to_csv("data/wellbeing.csv",index=False); print("Generated synthetic data/ wellbeing.csv")
