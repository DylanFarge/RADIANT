import pandas as pd
import numpy as np
from astroquery.skyview import SkyView
from zipfile import ZipFile
import io
import matplotlib.pyplot as plt
from astropy import units as u

# Load the data
data = np.array(pd.read_csv('check_pairs.csv', index_col=0, dtype=int))
dists = np.array(pd.read_csv('check_dists.csv', index_col=0))

# sort columns
data = np.sort(data, axis=1)

# sort rows by first column element
data = data[np.argsort(data[:,0])]

# Remove duplicates
print("Length before removing duplicates: ", len(data), " (", len(dists), ")")
data, indices = np.unique(data, axis=0, return_index=True)
dists = dists[indices]
print("Length after removing duplicates: ", len(data), " (", len(dists), ")")

# print("Pairs-n",data)
# print("Distances-n",dists)
full_list = np.hstack((data, dists))
np.set_printoptions(suppress = True) # Suppress scientific notation
full_list = full_list[full_list[:,2].argsort()]
print(full_list)

df_all = pd.read_json("src/database/official/df.json")
relevant_sources = np.unique(data.flatten())
print("Unique list of relevant sources->\n",relevant_sources)

# df = df_all.iloc[relevant_sources]
print("Relevant sources DataFrame->\n",df_all)

with ZipFile('images.zip', 'w') as zipf:

    # convert full_list to a csv and write it to zipf
    file = io.BytesIO()
    pd.DataFrame(full_list).to_csv(file, index=False, header=False)
    zipf.writestr("meta.csv", file.getvalue())

    for i, (s1, s2, distance) in enumerate(full_list):
        s1, s2 = int(s1), int(s2)

        print(f"---> Pair: {s1} and {s2}")
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

    # for i, entry in df.iterrows():
    #     print(f"Source {i}: {entry['RA/deg']} {entry['DEC/deg']}")
    #     try:
    #         print(f"\tTrying FIRST...", end="")
    #         fitsfile = SkyView.get_images(position=f"{entry['RA/deg']} {entry['DEC/deg']}", survey='VLA FIRST (1.4 GHz)', pixels=300)[0][0].data
    #         print("Success!")
    #     except:
    #         try:
    #             print(f"trying NVSS...", end="")
    #             fitsfile = SkyView.get_images(position=f"{entry['RA/deg']} {entry['DEC/deg']}", survey='NVSS', pixels=300)[0][0].data
    #             print("Success!")
    #         except:
    #             print("Failed")
    #             raise ValueError("No image found for source ",i," with coordinates ",entry['RA/deg'],entry['DEC/deg'])  
    #     plt.figure(frameon=False)
    #     plt.imshow(fitsfile, origin='lower', cmap='inferno')
    #     plt.axis('off')
    #     file = io.BytesIO()
    #     plt.savefig(file, format='png')        
    #     zipf.writestr(f"{i}_{entry['RA/deg']}_{entry['DEC/deg']}.png", file.getvalue())
    #     plt.close()

print("Images saved.")