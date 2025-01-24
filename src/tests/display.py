import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


with open("src/tests/saves/images.log") as f:
    df = pd.read_csv(f)

df = df[df["Images"] != "---"].reset_index(drop=True).astype({"Images": int, "Time": float})
epoch = 50

l = []
for i in range(len(df)):
    l.append((i//epoch) + 1)

df["Epoch"] = l

# fig, ax = plt.subplots(figsize=(7, 5))
# ft = 22
# plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
# ax.grid(True)
# ax = sns.lineplot(data=df, x="Images", y="Time")
# ax.set_xlabel("Number of Pictures", fontsize=ft)
# ax.set_ylabel("Time(s)", fontsize=ft)
# ax.tick_params(axis='both', labelsize=ft)
# fig.savefig("src/tests/saves/images.pdf")

# df = pd.read_csv("src/tests/saves/cumulative.log")
# fig, ax = plt.subplots(figsize=(7, 5))
# ft = 25
# plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
# ax.grid(True)
# ax = sns.lineplot(data=df, x="Number of Sources", y="Time Taken")
# ax.set_xlabel("Number of Sources", fontsize=ft)
# ax.set_ylabel("Time(s)", fontsize=ft)
# ax.tick_params(axis='both', labelsize=ft)
# fig.savefig("src/tests/saves/cumulative.pdf")

# df = pd.read_csv("src/tests/saves/cumulative_1000.log")
# fig, ax = plt.subplots(figsize=(7, 5))
# ft = 25
# plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
# ax.grid(True)
# ax = sns.lineplot(data=df, x="Number of Sources in Database", y="Time Taken", marker="o")
# ax.set_xlabel("Number of Sources", fontsize=ft)
# ax.set_xticks([20000,24000,28000])
# ax.set_ylabel("Time(s)", fontsize=ft)
# ax.tick_params(axis='both', labelsize=ft)
# fig.savefig("src/tests/saves/database.pdf")

# df = pd.read_csv("src/tests/saves/1_1000_50.log")
# fig, ax = plt.subplots(figsize=(7, 5))
# ft = 25
# plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
# ax.grid(True)
# ax = sns.lineplot(data=df, x="Number of Sources", y="Time Taken", marker="o")
# ax.set_xlabel("Number of Sources", fontsize=ft)
# ax.set_ylabel("Time(s)", fontsize=ft)
# ax.tick_params(axis='both', labelsize=ft)
# fig.savefig("src/tests/saves/new_cat.pdf")
