import base64
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from astropy.table import Table
import plotly.express as px
import pandas as pd
import numpy as np
import os, io, requests
import numexpr as ne
from astropy.io import fits
from astroquery.skyview import SkyView as sv
from matplotlib_venn import venn2 
from matplotlib import pyplot as plt
from dash import html
from flask_login import current_user
from zipfile import ZipFile
import seaborn as sns
from matplotlib.colors import LogNorm

class LOFAR: 
    Response = "Not_Active"
    start_url = "https://lofar-surveys.org/dr2_release.html"
    post_url = "https://lofar-surveys.org/dr2-cutout.fits"

def getImage(ra:float, dec:float, pixels:float, fov:float, survey:str, auto:bool):
    '''
    This function manages the retrieval of the radio wave image from the SkyView API.
    '''
    print(ra, dec, pixels, fov, survey, auto)
    coords = SkyCoord(ra, dec, unit=u.deg, frame='icrs')
    print("Getting Image")
    
    if survey == "LOFAR":
        # ---------------------------------------------------------------------------------------------
        #TODO: Incoperate pixels, fov and auto into the LOFAR scrape
        # Make initial GET request to retrieve CSRF token or other necessary parameters
        if LOFAR.Response == "Not_Active":
            print("First_time_ping")
            try:
                requests.get(LOFAR.start_url)
                LOFAR.Response = "Connection_Successful"
            except:
                print("Failed get request to LOFAR site")
                return None
        
        hms = coords.ra.hms
        dms = coords.dec.dms
        sign = "+" if dms.d >= 0 else "-"
        position = f"{int(hms.h)} {int(hms.m)} {int(hms.s)} {sign}{int(abs(dms.d))} {int(abs(dms.m))} {int(abs(dms.s))}"
        print(position)
        size = fov*60 if auto else pixels*0.025
        print(size)
        form_data = {'pos': position,'size': size}

        response = requests.post(LOFAR.post_url, data=form_data)
        
        if response.headers['Content-Type'] != 'application/fits':
            print('Not found---------------->')
            print(response.content.decode('utf-8').split('<h3>')[1].split('</h3>')[1].split('</div>')[0])
            print("------------------------")
            return None

        print('Found file')
        with fits.open(io.BytesIO(response.content)) as hdul:
            image = hdul[0].data
        # ---------------------------------------------------------------------------------------------
    else:
        flag = True
        while flag:
            flag = False
            try:
                if auto:
                    image = sv.get_images(position=coords, survey=[survey], radius=fov*u.deg, projection="Sin")[0][0].data
                else:
                    image = sv.get_images(position=coords, survey=[survey], pixels=pixels, projection="Sin")[0][0].data
            except TimeoutError:
                print("CAUGHT TIMEOUT ERROR - Trying again")
                flag = True
            except Exception as e:
                print(e)
                print("no image")
                return None
            
    plt.figure(frameon=False)
    plt.imshow(image,cmap='inferno', origin='lower')      
    plt.axis('off')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png',bbox_inches='tight', pad_inches=0)
    image_png = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    picture = f"data:image/png;base64,{image_png}"

    print("Got Image")
    return picture

def analysis_colours(get=False, set=False, update=False, df=None):
    user = current_user
    if set == True:
        grouped = df.groupby(["Type","Catalog"]).count().reset_index(level=[0,1])
        grouped.rename(columns={"RA/deg":"Number of sources"}, inplace=True)
        graph = px.bar(grouped, x="Type", y="Number of sources", color="Catalog")
        cs = {trace.name: trace.marker.color for trace in graph.data}
        return cs
    
    elif update == True:
        grouped = user.df.groupby(["Type","Catalog"]).count().reset_index(level=[0,1])
        grouped.rename(columns={"RA/deg":"Number of sources"}, inplace=True)
        graph = px.bar(grouped, x="Type", y="Number of sources", color="Catalog")
        cs = {trace.name: trace.marker.color for trace in graph.data}
        user.colours = cs

    elif get:
        if user.colours == None:
            raise ValueError("user.colours was NONE and not before set.")
        return user.colours
    
    else:
        raise ValueError("Must specify either get or set")
    

# def setup():
#     '''
#     Reads in all the data fro`m the catalogs folder and stores it in a dataframe
#     ''' 
#     global df, colours
#     print("Setting up data...")
    
#     # Read in the preset catalogs
#     data = []

#     # Due to layout of file 'an.json', read in order of:
#     for file in ["FRI","FRII","unLRG","MiraBest","LRG","Proctor","FR0"]:
#         data.append(Table.read("catalogs/"+file+".fits")) 
            
#     # Read in the uploaded catalogs
#     if not os.path.exists("catalogs/uploaded"):
#         os.mkdir("catalogs/uploaded")
#     for file in os.listdir("catalogs/uploaded"):
#         if 'fits' in file:
#             data.append(Table.read("catalogs/uploaded/"+file))

#     # Store the data in a dataframe
#     for cat in data:
#         h = [int(i) for i in cat["RAh"].tolist()]
#         m = [int(i) for i in cat["RAm"].tolist()]
#         s = [float(i) for i in cat["RAs"].tolist()]
#         ra_hms = [f"{h}h{m}m{s}s" for h,m,s in zip(h,m,s)]

#         sn = [i for i in cat["DECsign"].tolist()]
#         d = [int(i) for i in cat["DECd"].tolist()]
#         m = [int(i) for i in cat["DECm"].tolist()]
#         s = [float(i) for i in cat["DECs"].tolist()]
#         dec_dms = [f"{sn}{d}d{m}m{s}s" for sn,d,m,s in zip(sn,d,m,s)]
        
#         sk = SkyCoord(ra_hms,dec_dms)
        
#         df = pd.concat([df,pd.DataFrame({
#             'RA/deg':sk.ra.degree,
#             'DEC/deg':sk.dec.degree,
#             'Catalog':cat["Catalog"].tolist(),
#             'Type':cat["Type"].tolist(),
#         })], ignore_index=True)

        
#         setColours()
#     print("...Data setup complete")
    
                
def getSunburst():
    '''
    Creates and returns a sunburst plot of the catalogs' compositions as a plotly figure
    '''
    df = current_user.df
    count_df = pd.DataFrame(columns=['Catalog', 'Type', 'Count'])
    
    for catalog in df["Catalog"].unique():
        for t in df["Type"].unique():
            
            count = len(df[(df["Catalog"] == catalog) & (df["Type"] == t)])
            new_df = pd.DataFrame({'Catalog': catalog,'Type': t,'Count': count}, index=[0])
            count_df = pd.concat([count_df, new_df])
    try:
        fig = px.sunburst(count_df, path=['Catalog', 'Type'], values='Count', template='plotly_dark', color='Catalog', color_discrete_map=analysis_colours(get=True))
    except:
        fig = px.sunburst(count_df, path=['Catalog', 'Type'], values='Count', template='plotly_dark', color='Catalog')
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    return fig


def get_surveys():
    '''
    Returns a list of the surveys
    '''
    surveys= {
        "LOFAR": "LOFAR",
        "VLA FIRST (1.4 GHz)": "VLA FIRST (1.4 GHz)",
        "NVSS": "NVSS",
        "DSS": "DSS (not radio)",
        # 'CO': 'CO',
        # 'GB6 (4850MHz)': 'GB6',
        # 'Stripe82VLA': 'Stripe82VLA',
        # '1420MHz (Bonn)': 'Bonn',
        # 'HI4PI': 'HI4PI',
        # 'EBHIS': 'EBHIS',
        # 'nH': 'nH',
        'GLEAM 72-103 MHz': 'GLEAM 72-103 MHz',
        'GLEAM 103-134 MHz': 'GLEAM 103-134 MHz',
        'GLEAM 139-170 MHz': 'GLEAM 139-170 MHz',
        'GLEAM 170-231 MHz': 'GLEAM 170-231 MHz',
        # 'SUMSS 843 MHz': 'SUMSS 843 MHz',
        # '0408MHz': '0408MHz',
        # 'WENSS': 'WENSS',
        'TGSS ADR1': 'TGSS ADR1',
        'VLSSr': 'VLSSr',
        '0035MHz': '0035MHz',
    }
    return [{"label": html.Span(k,style={'color': '#08F',}), "value": v} for k,v in surveys.items()]


def getCatalogs(list: list[str], types :list[str]) -> pd.DataFrame:
    '''
    Returns a dataframe of the catalogs in the list
    '''
    df = current_user.df
    mask1 = df["Catalog"].isin(list)
    mask2 = df["Type"].isin(types)
    mask = mask1 & mask2
    return df[mask]

def get_catalog_names() -> list[str]:
    return current_user.df["Catalog"].unique().tolist()

def get_morphology_names(catalogs = None) -> list[str]:
    if catalogs:
        return current_user.df[current_user.df["Catalog"].isin(catalogs)]["Type"].unique().tolist()
    return []
        
def sphere_dist(ra1, dec1, ra2, dec2):
    ra1 = np.radians(ra1).astype(np.float64)
    ra2 = np.radians(ra2).astype(np.float64)
    dec1 = np.radians(dec1).astype(np.float64)
    dec2 = np.radians(dec2).astype(np.float64)

    numerator = ne.evaluate('sin((dec2 - dec1) / 2) ** 2 + cos(dec1) * cos(dec2) * sin((ra2 - ra1) / 2) ** 2')

    dists = ne.evaluate('2 * arctan2(numerator ** 0.5, (1 - numerator) ** 0.5)')
    return np.degrees(dists) 
    
def analyse_new_catalog(new_cat_name: str):
    df = current_user.df
    an = current_user.an
    # other_names = df["Catalog"].unique().tolist()[:-1]
    stored_cats_names = an.columns.to_list()
    new_cat = df[df["Catalog"] == new_cat_name]
    
    #Insert new column
    print("Inserting new column:",new_cat_name)
    insert_col = []
    for cat_name in stored_cats_names:
        index_of_sources = df[df["Catalog"] == cat_name].index.to_list()
        insert_col.append([[180, i, -1] for i in index_of_sources])
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
        dists[from_index - start_offset] = 180 + 1 # Ignore self
        minimals.append([dists.min(), from_index, dists.argmin() + start_offset])

    insert_row.append(minimals)
        
    # Join the new row to the analysis
    print("Joining...")
    an.loc[new_cat_name] = insert_row

    # Sort all of the arrays in the dataframe
    print("Sorting...")
    for i in an.index:
        for j in an.columns:
            an.at[i,j].sort()
    print("...done")
        
    print("Saving...")
    an.to_json("src/database/"+current_user.id+"/analysis.json")
    df.to_json("src/database/"+current_user.id+"/df.json")
    current_user.an = an
    current_user.df = df
    

def get_closest_dist(catalog: str):
    an = current_user.an
    lines = an.loc[catalog]
    lines_names = lines.index.tolist()
    line_arr = np.array(lines.to_list())[:,:,0].T
    return pd.DataFrame(line_arr, columns=lines_names)

    
def download(set_progress, cats: list[str], types: list[str], hasImage, surveys, image_limit, threshold, prior):
    set_progress((str(20), "Starting Process", str(100)))
    df = current_user.df
    data = df[df["Catalog"].isin(cats) & df["Type"].isin(types)]
    if threshold != None and threshold != 0:
        set_progress((str(100), "Removing Duplicates", str(100)))
        remove = remove_duplicates(cats, threshold, prior)
        data = data[~data.index.isin(remove)]
    data.sort_values(by=["RA/deg", "DEC/deg"], inplace=True)
    coords = SkyCoord(data["RA/deg"], data["DEC/deg"], unit="deg")
    hh, hm, hs = coords.ra.hms
    dd, dm, ds = coords.dec.dms
    new_df = pd.DataFrame({
        "RAh": hh,
        "RAm": hm,
        "RAs": hs,
        "DECsign": ["+" if d >= 0 and d2 >= 0 and d3 >= 0 else "-" for d,d2,d3 in zip(dd,dm,ds)],
        "DECd": [abs(d) for d in dd],
        "DECm": [abs(d) for d in dm],
        "DECs": [abs(d) for d in ds],
        "Catalog": data["Catalog"],
        "Type": data["Type"],
    }) 
    
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w') as zipf:
        
        zipf.writestr('RADCAT.csv', new_df.to_csv(index=False))
    
        if image_limit == None:
            image_limit = len(coords)
            max_progress = len(coords) 
        else:
            max_progress = min(image_limit, len(coords) * len(surveys))
            
        if hasImage:
            counter = 0
            for coord in coords:
                ra_deg = coord.ra.deg
                dec_deg = coord.dec.deg
                sign = "+" if dec_deg >=0 else ""
                name=f"J{ra_deg}{sign}{dec_deg}"
                for survey in surveys:
                    if survey == "LOFAR":
                        # Make initial GET request to retrieve CSRF token or other necessary parameters
                        if LOFAR.Response == "Not_Active":
                            print("First_time_ping")
                            try:
                                requests.get(LOFAR.start_url)
                                LOFAR.Response = "Connection_Successful"
                            except:
                                print("Failed get request to LOFAR site")
                                return None
                        
                        hms = coord.ra.hms
                        dms = coord.dec.dms
                        sign = "+" if dms.d >= 0 else "-"
                        position = f"{int(hms.h)} {int(hms.m)} {int(hms.s)} {sign}{int(abs(dms.d))} {int(abs(dms.m))} {int(abs(dms.s))}"
                        print(position)
                        
                        form_data = {'pos': position,'size': str(0.025*50)}

                        response = requests.post(LOFAR.post_url, data=form_data)
                        
                        if response.headers['Content-Type'] != 'application/fits':
                            print(f"Could not download image of {name} for {survey}") 
                            continue

                        print('Found file')
                        found_file = io.BytesIO()
                        with fits.open(io.BytesIO(response.content)) as hdul:
                            hdul[0].writeto(found_file)
                    else:
                        flag = True
                        while flag:
                            flag = False
                            try:
                                images = sv.get_images(position=coord, survey=survey, cache=True)
                            except TimeoutError:
                                print("CAUGHT TIMEOUT ERROR - Trying again")
                                flag = True
                            except:
                                print(f"Could not download image of {name} for {survey}") 
                                continue
                        found_file = io.BytesIO()
                        images[0].writeto(found_file)
                    
                    zipf.writestr(name+"_"+survey+".fits", found_file.getvalue())
                    
                    counter += 1
                    progress = int((counter/max_progress )* 100)
                    set_progress((str(progress), "Downloading: "+str(progress)+"%", str(100)))
                    if image_limit == counter:
                        break
                if image_limit == counter:
                    break

    set_progress((str(0), "", str(100)))
    return buffer.getvalue()
    

def newCatalog(dataframe, name,raName,decName, types):
    df = current_user.df
    new = pd.DataFrame({"Index":np.linspace(0,len(dataframe)-1, len(dataframe), dtype=int),"RA/deg":dataframe[raName],"DEC/deg":dataframe[decName],"Catalog":name, "Type":dataframe[types]})
    # change dtype of RA/deg and DEC/deg to float
    df = pd.concat([df,new], ignore_index=True)
    current_user.df = df
    ra= Angle(dataframe[raName], unit="deg").hms
    dec = Angle(dataframe[decName], unit="deg").dms
    sign = ["-" if "-" in str(dataframe[decName][i]) else "+" for i in range(len(dataframe[decName]))]
    new = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign ,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":name, "Type":dataframe[types]})
    
    for i in range(len(new)):
        new["RAh"][i] = int(new["RAh"][i])
        new["RAm"][i] = int(new["RAm"][i])
        new["DECd"][i] = np.abs(int(new["DECd"][i]))
        new["DECm"][i] = np.abs(int(new["DECm"][i]))
        new["DECs"][i] = np.abs(new["DECs"][i])

    new = new.sort_values(by=['RAh','RAm','RAs'])
    t = Table.from_pandas(new)
    # t.write("catalogs/uploaded/"+name+".fits", format = 'fits')
    t.write(f"src/database/{current_user.id}/{name}.fits", format = 'fits')
    analysis_colours(update=True)
    print("I GOT TO A NEW CAT!")
    
    
def rmCatalog(cats_rm):
    df = current_user.df
    an = current_user.an 
    df.drop(df[df["Catalog"].isin(cats_rm)].index, inplace=True, )
    df.reset_index(inplace=True, drop=True)
    an.drop(cats_rm, inplace=True, axis=1)
    an.drop(cats_rm, inplace=True, axis=0)
    [os.remove(f"src/database/{current_user.id}/"+rm_cat+".fits") for rm_cat in cats_rm]
    an.to_json(f"src/database/{current_user.id}/analysis.json")
    df.to_json(f"src/database/{current_user.id}/df.json")
    current_user.df = df
    current_user.an = an


def getTypesGraph():
    '''
    Returns a px bar graph of the types of galaxies composed from the different catalogs.
    '''
    df = current_user.df
    grouped = df.groupby(["Type","Catalog"]).count().reset_index(level=[0,1])
    grouped.rename(columns={"RA/deg":"Number of sources"}, inplace=True)
    try:
        fig = px.bar(grouped, x="Type", y="Number of sources", color="Catalog", color_discrete_map=analysis_colours(get=True), template="plotly_dark")
    except:
        fig = px.bar(grouped, x="Type", y="Number of sources", color="Catalog", template="plotly_dark")
    fig.update_layout(title_text="Morphology composition by catalog", title_x=0.5, paper_bgcolor="rgb(0,0,0,0)")
    return fig


def remove_duplicates(names, threshold, prior:list):
    an = current_user.an
    print("Begin Graph Calc...")
    #* Creating the edges dictionary
    edges = {}
    removed = set()
    for i in names:
        if i in prior:
            priority = prior.index(i)
            if priority == 0:
                continue
            some_names = prior[:priority]
            is_prior = True
        else:
            some_names = names
            is_prior = False
        for j in some_names:
            sources = np.array(an.at[i, j])
            print(len(sources[sources[:,0] <= threshold]))
            if is_prior or j in prior:
                verticies = sources[sources[:,0] <= threshold][:,1].astype(int)
                for v in verticies:
                    removed.add(v)
                continue
            pairs = sources[sources[:,0] <= threshold][:,1:].astype(int)
            print(f"pairs: {len(pairs)}")
            if pairs.size > 0:
                for f,t in pairs:
                    f = int(f)
                    t = int(t)
                    if f in edges:
                        edges[f].add(t)
                    else:
                        edges[f] = set([t])
                        
                    if t in edges:
                        edges[t].add(f)
                    else:
                        edges[t] = set([f])

    #* Finding the vertex with the most edges and removing them
    print(f"Len of edges: {len(edges)}")
    while edges != {}:
        max_vertex = max(edges, key=lambda k: len(edges[k]))
        for v in edges[max_vertex]:
            edges[v].remove(max_vertex)
            if len(edges[v]) == 0:
                del edges[v]
        del edges[max_vertex]
        removed.add(max_vertex)
    print("...End Graph Calc")
    return removed

def simulate_thresholds(set_progress, start, finish, step, cat):
    print("SIMULATING HERE")
    set_progress((str(20), "Starting Process", str(100)))
    an = current_user.an
    if cat != "All Catalogs":
        an = an.loc[:, [cat]]
    thresholds = np.arange(start, finish, step)
    results = []
    names_col = an.columns.tolist()
    names_row = an.index.tolist()
    edges = {}
    for counter, threshold in enumerate(thresholds):
        progress = int(counter/len(thresholds) * 100)
        set_progress((str(progress), f"{progress}%", str(100)))
        #* Creating the edges dictionary
        for i in names_row:
            for j in names_col:
                sources = np.array(an.at[i, j])
                pairs = sources[sources[:,0] <= threshold][:,1:].astype(int)
                if pairs.size > 0:
                    for f,t in pairs:
                        if f in edges:
                            edges[f].add(t)
                        else:
                            edges[f] = set([t])
                            
                        if t in edges:
                            edges[t].add(f)
                        else:
                            edges[t] = set([f])

        #* Finding the vertex with the most edges and removing them
        removed = 0
        while edges != {}:
            max_vertex = max(edges, key=lambda k: len(edges[k]))
            for v in edges[max_vertex]:
                edges[v].remove(max_vertex)
                if len(edges[v]) == 0:
                    del edges[v]
            del edges[max_vertex]
            removed += 1
            
        results.append(removed)
    fig = px.line(y=results, x=thresholds ,template="plotly_dark", title="Number of duplicates detected with change in threshold across all catalogs")
    fig.update_xaxes(title_text="Distance thresholds in degrees")
    fig.update_yaxes(title_text="Number of duplicates detected")
    fig.update_layout(paper_bgcolor="rgb(0,0,0,0)", title_x=0.5)
    set_progress((str(0), "", str(100)))
    return fig


def overlap_heatmap(threshold, log):
    an = current_user.an
    fig, ax = plt.subplots(figsize=(6,5))
    fig.patch.set_facecolor('none')
    names = []
    ncols = len(an.columns)
    print(ncols)
    hmap = np.zeros((ncols, ncols))

    relevant_pairs_len = 0 # XXX
    relevant_pairs = [] # XXX
    relevant_dists = [] # XXX

    for i, (name, data) in enumerate(an.items()):
        names.append(name)
        for j, row in enumerate(data):
            nprow = np.array(row)
            hmap[j,i] = len(nprow[nprow[:,0] <= threshold])

            # onerqed = nprow[nprow[:,0] <= 0.017] # XXX
            # pairs = onerqed[onerqed[:,0] >= 0.014][:,1:] # XXX
            # onerqed = nprow[nprow[:,0] <= 0.019] # XXX
            # pairs = onerqed[onerqed[:,0] > 0.017][:,1:] # XXX
            # onerqed = nprow[nprow[:,0] <= 0.021] # XXX
            # pairs = onerqed[onerqed[:,0] > 0.019][:,1:] # XXX
            # onerqed = nprow[nprow[:,0] <= 0.050] # XXX
            # pairs = onerqed[onerqed[:,0] > 0.040][:,1:] # XXX
            onerqed = nprow[nprow[:,0] <= 0.050] # XXX
            onerqed = onerqed[onerqed[:,0] > 0.016] # XXX
            pairs = onerqed[:,1:] # XXX
            dists = onerqed[:,0] # XXX
            relevant_pairs_len += len(pairs) # XXX
            if len(pairs) > 0: # XXX
                relevant_pairs.extend(pairs.tolist()) # XXX
                relevant_dists.extend(dists.tolist())

    # print(f"Relevant pairs: {relevant_pairs}") # XXX
    print(f"Relevant length: {relevant_pairs_len}") # XXX
    # pd.DataFrame(relevant_pairs).to_csv("check_pairs.csv") # XXX
    # pd.DataFrame(relevant_dists).to_csv("check_dists.csv") # XXX


    heat_df = pd.DataFrame(hmap, columns=names, index=names)
    cmap = plt.colormaps.get_cmap("viridis")
    cmap.set_bad('black')
    if log:
        sns.heatmap(heat_df, annot=False, cmap=cmap, ax=ax, norm=LogNorm())
    else:
        sns.heatmap(heat_df, annot=False, cmap=cmap, ax=ax)
    for  label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color('white')
    cbar= ax.collections[0].colorbar
    cbar.ax.tick_params(colors='white')
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png',bbox_inches='tight', pad_inches=0)
    image_png = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    picture = f"data:image/png;base64,{image_png}"

    return html.Img(
        src=picture,
        style={
            'height': '50%',
            'margin':'auto',
        }
    )


def overlap_venn(threshold, ven1, ven2):
    an = current_user.an
    df = current_user.df
    fig = plt.figure(figsize=(5,5))
    fig.patch.set_facecolor('none')

    sources1 = np.array(an.at[ven1, ven2])
    sources2 = np.array(an.at[ven2, ven1])

    overlay1 = set()
    overlay2 = set()
    for i in sources1[sources1[:,0] <= threshold]:
        overlay1.add(i[1])
        overlay2.add(i[2])
        if i[0] > 0.015:
            s1 = int(i[1])
            s2 = int(i[2])
            print("<<Source 1>>")
            print("RA DEC: ", df.at[s1, "RA/deg"], df.at[s1, "DEC/deg"])
            print("Type: ", df.at[s1, "Type"])
            print("<<Source 2>>")
            print("RA DEC: ", df.at[s2, "RA/deg"], df.at[s2, "DEC/deg"])
            print("Type: ", df.at[s2, "Type"])
            print(f"___<<Distance {i[0]}>>___")

    print("---")

    for i in sources2[sources2[:,0] <= threshold]:
        overlay2.add(i[1])
        overlay1.add(i[2])

    mini = min(len(overlay1), len(overlay2))

    if ven1 == ven2:
        mini = mini // 2

    numbers = [
        len(sources1) - mini,
        len(sources2) - mini,
        mini
    ]

    print(f"Found {numbers[2]} overlapping sources")
    v = venn2(numbers, set_labels = (ven1, ven2), set_colors=('lightblue', 'm'))
    
    for label in v.subset_labels:
        label.set_color('white') if label else None
    for text in v.set_labels:
        text.set_color('white') if text else None

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png',bbox_inches='tight', pad_inches=0)
    image_png = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    picture = f"data:image/png;base64,{image_png}"

    print(len(overlay1), len(overlay2))

    return html.Img(
        src=picture,
        style={
            'height': '50%',
            'margin':'auto',
        }
    ), [mini/len(sources1), mini/len(sources2)],



    
