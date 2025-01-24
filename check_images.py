import pandas as pd
import numpy as np
from zipfile import ZipFile
from astroquery.skyview import SkyView
import matplotlib.pyplot as plt
import io
from astropy import units as u


DF_ANALYSIS_PATH = 'src/database/official/'
LOWER, UPPER = 0.025 , 0.050 

an = pd.read_json(DF_ANALYSIS_PATH + 'analysis.json')

relevant_pairs = []

for data_col in an.values:
    for data in data_col:
        data = np.array(data)
        cut = data[(data[:,0] > LOWER) & (data[:,0] <= UPPER)]
        relevant_pairs.extend(cut.tolist())
relevant_pairs = np.array(relevant_pairs)
print("---Started---")
print("Distance Bracket",LOWER,"-",UPPER)
relevant_pairs = relevant_pairs[np.argsort(relevant_pairs[:,0])]

dists, idxs = relevant_pairs[:,0].reshape((-1,1)), relevant_pairs[:,1:]
maxis = np.max(idxs, axis=1).reshape((-1,1))
minis = np.min(idxs, axis=1).reshape((-1,1))
relevant_pairs = np.hstack([dists, minis, maxis])
print("Relevant Number of Pairs: ", len(relevant_pairs), end=" ")
relevant_pairs = np.unique(relevant_pairs, axis=0)
print(f"({len(relevant_pairs)} unique)")
np.set_printoptions(suppress = True) # Suppress scientific notation

df_all = pd.read_json("src/database/official/df.json")

with ZipFile('images.zip', 'w') as zipf:

    # convert full_list to a csv and write it to zipf
    file = io.BytesIO()
    pd.DataFrame(relevant_pairs, columns=["Distances","Source 1", "Source 2"]).to_csv(file)
    zipf.writestr("meta.csv", file.getvalue())

    for i, (distance, s1, s2) in enumerate(relevant_pairs):
        s1, s2 = int(s1), int(s2)

        print(f"({i+1}/{len(relevant_pairs)})---> Pair: {s1} and {s2}")
        df_s1, df_s2 = df_all.iloc[s1], df_all.iloc[s2]

        print("\tTrying FIRST...", end="")
        first_1 = SkyView.get_images(position=f"{df_s1['RA/deg']} {df_s1['DEC/deg']}", survey='VLA FIRST (1.4 GHz)', radius=0.15*u.deg)[0][0].data
        first_2 = SkyView.get_images(position=f"{df_s2['RA/deg']} {df_s2['DEC/deg']}", survey='VLA FIRST (1.4 GHz)', radius=0.15*u.deg)[0][0].data
        print("Success!")

        print("\tTrying NVSS...", end="")
        nvss_1 = SkyView.get_images(position=f"{df_s1['RA/deg']} {df_s1['DEC/deg']}", survey='NVSS', radius=0.15*u.deg)[0][0].data
        nvss_2 = SkyView.get_images(position=f"{df_s2['RA/deg']} {df_s2['DEC/deg']}", survey='NVSS', radius=0.15*u.deg)[0][0].data
        print("Success!")

        print("<<<---Saving images...", end="")
        fig, ax = plt.subplots(2, 2, figsize=(15, 15))
        
        ax[0, 0].imshow(first_1, origin='lower', cmap='inferno')
        ax[0, 0].axis('off')
        ax[0, 0].set_title(f"FIRST {s1}: {df_s1['RA/deg']} {df_s1['DEC/deg']}", color='white')
        ax[0, 0].plot(first_1.shape[1] // 2, first_1.shape[0] // 2, 'go', markersize=2)

        ax[0, 1].imshow(first_2, origin='lower', cmap='inferno')
        ax[0, 1].axis('off')
        ax[0, 1].set_title(f"FIRST {s2}: {df_s2['RA/deg']} {df_s2['DEC/deg']}", color='white')
        ax[0, 1].plot(first_2.shape[1] // 2, first_2.shape[0] // 2, 'go', markersize=2)

        ax[1, 0].imshow(nvss_1, origin='lower', cmap='inferno')
        ax[1, 0].axis('off')
        ax[1, 0].set_title(f"NVSS {s1}: {df_s1['RA/deg']} {df_s1['DEC/deg']}", color='white')
        ax[1, 0].plot(nvss_1.shape[1] // 2, nvss_1.shape[0] // 2, 'go', markersize=2)

        ax[1, 1].imshow(nvss_2, origin='lower', cmap='inferno')
        ax[1, 1].axis('off')
        ax[1, 1].set_title(f"NVSS {s2}: {df_s2['RA/deg']} {df_s2['DEC/deg']}", color='white')
        ax[1, 1].plot(nvss_2.shape[1] // 2, nvss_2.shape[0] // 2, 'go', markersize=2)

        file = io.BytesIO()
        fig.tight_layout()
        fig.patch.set_facecolor('black')
        fig.savefig(file, format='png')
        zipf.writestr(f"{i}_{distance}_{s1}_{s2}.png", file.getvalue())
        plt.close()
        print("Success!--->>>")