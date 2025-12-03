import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ablation_summary.csv")

df["best_val_acc"] = pd.to_numeric(df["best_val_acc"], errors="coerce")
df["params"] = pd.to_numeric(df["params"], errors="coerce")

plt.figure(figsize=(18,5))
sns.barplot(data=df, x="experiment", y="best_val_acc")
plt.xticks(rotation=90)
plt.title("Validation Accuracy for All Ablation Experiments")
plt.tight_layout()
plt.savefig("acc_bar.png")
plt.close()

plt.figure(figsize=(18,5))
sns.barplot(data=df, x="experiment", y="params")
plt.xticks(rotation=90)
plt.title("Parameter Count for All Ablation Experiments")
plt.tight_layout()
plt.savefig("param_bar.png")
plt.close()

def avg_acc(cond):
    return df[cond]["best_val_acc"].mean()

def avg_params(cond):
    return df[cond]["params"].mean()

records = []

records.append(["BN", 
    avg_acc(df.use_bn == 1) - avg_acc(df.use_bn == 0),
    avg_params(df.use_bn == 1) - avg_params(df.use_bn == 0)
])

records.append(["Dropout", 
    avg_acc(df.use_dropout == 1) - avg_acc(df.use_dropout == 0),
    avg_params(df.use_dropout == 1) - avg_params(df.use_dropout == 0)
])

if "use_residual" in df.columns:
    records.append(["Residual",
        avg_acc(df.use_residual == 1) - avg_acc(df.use_residual == 0),
        avg_params(df.use_residual == 1) - avg_params(df.use_residual == 0)
    ])

if "pooling" in df.columns:
    for ptype in ["max","avg","stride","none"]:
        cond = (df.pooling == ptype)
        records.append([f"Pooling_{ptype}",
            avg_acc(cond) - df["best_val_acc"].mean(),
            avg_params(cond) - df["params"].mean()
        ])

delta_df = pd.DataFrame(records, columns=["Technique", "Delta_Acc", "Delta_Params"])

plt.figure(figsize=(10,6))
sns.barplot(data=delta_df, x="Technique", y="Delta_Acc")
plt.xticks(rotation=45)
plt.title("ΔAccuracy for Different Ablations")
plt.tight_layout()
plt.savefig("delta_accuracy.png")
plt.close()

plt.figure(figsize=(10,6))
sns.barplot(data=delta_df, x="Technique", y="Delta_Params")
plt.xticks(rotation=45)
plt.title("ΔParams for Different Ablations")
plt.tight_layout()
plt.savefig("delta_params.png")
plt.close()

print("Generated acc_bar.png, param_bar.png, delta_accuracy.png, delta_params.png")
