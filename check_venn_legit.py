import pandas as pd

df = pd.read_json("src/database/official/df.json")

cats = [
    # ["LRG", "GKCBR23"],
    # ["FRGMRC", "SGBW22"],
    ["CoNFIG", "CoNFIG"],
    # ["MiraBest", "CoNFIG"],
]

for cat1, cat2 in cats:
    df1 = df[df["Catalog"] == cat1]
    df2 = df[df["Catalog"] == cat2]
    for i, row1 in df1.iterrows():
        for j, row2 in df2.iterrows():
            if cat1 == cat2 and i == j:
                continue
            if row1["RA/deg"] == row2["RA/deg"] and row1["DEC/deg"] == row2["DEC/deg"]:
                print(f"Found a match...: {row1['RA/deg']}, {row1['DEC/deg']},-> {cat1}")
                print(f".............and: {row2['RA/deg']}, {row2['DEC/deg']},-> {cat2}")
