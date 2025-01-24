import os
import pandas as pd
from astropy import coordinates, table
from processing.data import sphere_dist
import processing.createFits as createFits
from assets.user import Credentials

# ------------------ Global Settings ------------------
print("-- Using devTools ---")
ANALYSIS_PATH = "src/processing/devOut/analysis.json"
DATAFRAME_PATH = "src/processing/devOut/df.json"
OFFICIAL_FITS_PATH = "src/processing/devOut/"
MAX_DEG = 180

# ------------------ Private Functions ------------------
def _construct_df():
    print("Reading CATS\nProcessing...",)        
    df = pd.DataFrame()
    for file in sorted(os.listdir(OFFICIAL_FITS_PATH)):
        print(f"...{file}")
        if file.endswith(".fits"):
            
            cat = table.Table.read(OFFICIAL_FITS_PATH + file, format="fits")
            
            h = [int(i) for i in cat["RAh"].tolist()]
            m = [int(i) for i in cat["RAm"].tolist()]
            s = [float(i) for i in cat["RAs"].tolist()]
            ra_hms = [f"{h}h{m}m{s}s" for h,m,s in zip(h,m,s)]

            sn = [i for i in cat["DECsign"].tolist()]
            d = [int(i) for i in cat["DECd"].tolist()]
            m = [int(i) for i in cat["DECm"].tolist()]
            s = [float(i) for i in cat["DECs"].tolist()]
            dec_dms = [f"{sn}{d}d{m}m{s}s" for sn,d,m,s in zip(sn,d,m,s)]

            if file == "CoNFIG.fits":
                print("Noted that original CoNFIG has a dec containing '60.0' seconds. It's fine.")                        

            sk = coordinates.SkyCoord(ra_hms,dec_dms)
        
            df = pd.concat([df,pd.DataFrame({
                'RA/deg':sk.ra.degree,
                'DEC/deg':sk.dec.degree,
                # 'SkyCoord':sk,
                'Catalog':cat["Catalog"].tolist(),
                'Type':cat["Type"].tolist(),
            })], ignore_index=True)

    df.sort_values(by=['Catalog','RA/deg','DEC/deg'], inplace=True, ignore_index=True)
    print('Saving...')
    df.to_json(DATAFRAME_PATH)

def _get_or_create_df():
    if os.path.exists(DATAFRAME_PATH) == False:
        _construct_df()
    return pd.read_json(DATAFRAME_PATH)

def _get_an():
    if os.path.exists(ANALYSIS_PATH) == False:
        with open(ANALYSIS_PATH, "w") as f: f.write("{}")
    return pd.read_json(ANALYSIS_PATH)

# ------------------ Public Functions ------------------
def override_dataframe_build():
    _construct_df()

def override_analysis_build():
    df = _get_or_create_df()
    an = _get_an()

    stored_cats_names = an.columns.to_list()

    new_cats_names = [cat for cat in df["Catalog"].unique().tolist() if cat not in stored_cats_names]
    
    for new_cat_name in new_cats_names:

        new_cat = df[df["Catalog"] == new_cat_name]

        #Insert new column
        print("Inserting new column:",new_cat_name)
        insert_col = []
        for cat_name in stored_cats_names:
            index_of_sources = df[df["Catalog"] == cat_name].index.to_list()
            insert_col.append([[MAX_DEG, i, -1] for i in index_of_sources])
        
        an[new_cat_name] = insert_col

        # Build new row and simultaneously update inserted column
        print("Comparing...")
        insert_row = []
        for cat_name in stored_cats_names:

            minimals = []
            cat = df[df["Catalog"] == cat_name]
            start_offset = cat.index.to_list()[0]

            for name, source in new_cat.iterrows():
                to_index = name

                dists = sphere_dist(source["RA/deg"], source["DEC/deg"], cat["RA/deg"], cat["DEC/deg"])

                # Compare distances
                cached_row_element = an.at[cat_name, new_cat_name]

                for i, (stored_dist, from_index, _ ) in enumerate(cached_row_element):
                    dist_offset = from_index - start_offset
                    if dists[dist_offset] < stored_dist:
                        an.at[cat_name, new_cat_name][i] = [dists[dist_offset], from_index, to_index]

                from_index = to_index # Updating the new row elements
                minimals.append([dists.min(), from_index, dists.argmin() + start_offset])

            insert_row.append(minimals)

        # Internal duplicates are calculated a slightly different way
        print("Calculating internal potential duplicates...")
        start_offset = new_cat.index.to_list()[0]
        minimals = []

        for from_index, source in new_cat.iterrows():

            dists = sphere_dist(source["RA/deg"], source["DEC/deg"], new_cat["RA/deg"], new_cat["DEC/deg"])
            try:
                dists[from_index - start_offset] = MAX_DEG + 1 # Ignore self
            except:
                print(new_cat)
                raise Exception("Error with",from_index, start_offset, len(dists), len(new_cat), new_cat_name)
            minimals.append([dists.min(), from_index, dists.argmin() + start_offset])

        insert_row.append(minimals)

        # Join the new row to the analysis
        print("Joining...")
        an.loc[new_cat_name] = insert_row

        stored_cats_names.append(new_cat_name)

    # Sort new entries to make it easier to read
    if new_cats_names != []:
        print("Sorting...")
        for new_catalog in new_cats_names:
            for catalog in stored_cats_names:
                an.at[catalog, new_catalog].sort()
                an.at[new_catalog, catalog].sort()

    print("Saving...")
    an.to_json(ANALYSIS_PATH)
    

def create_fits(args):
    if args == []:
        print("No arguments provided...stopping")
        return
    
    catalogs = [arg.lower() for arg in args]
    
    functions = {}
    for key in [func for func in  dir(createFits) if func.startswith("create_")] :
        functions[key[7:].lower()] = getattr(createFits, key)
        
    if catalogs[0] == "all":
        catalogs = functions.keys()
    else:
        for cat in catalogs:
            if cat not in functions.keys():
                print(f"Catalog {cat} not found...stopping")
                return
        
    for cat in catalogs:
        print(f"Creating {cat}...")
        functions[cat]()


def remove_user(users):
    if users == []:
        print("No arguments provided...stopping")
        return
    
    for user in users:
        try:
            Credentials.delete_account(user)
        except:
            print("************* Error in deleting account user",user)
