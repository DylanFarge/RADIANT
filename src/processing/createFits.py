import astropy.io.fits as fits
import pandas as pd
from astropy.table import Table
from astropy.coordinates import Angle
import os
import numpy as np
from astroquery.simbad import Simbad

OUTPUT_PATH = "src/processing/devOut/"
    
#* Total Sources = 960
def create_FRGMRC():
    df_raw = pd.read_csv("doc/catalogs/FRGMRC/FRGMRC-221022-RELEASE-V1.0/FRGMRC-221022-RELEASE-V1.0.csv")
    
    morphs = df_raw['NAME'].apply(lambda x: x.split('_')[0])
    ra = Angle(df_raw['RA'], unit='deg').hms
    dec = Angle(df_raw['DEC'].apply(lambda x: abs(x)), unit='deg').dms
    sign = ["-" if "-" in str(x) else "+" for x in df_raw['DEC']]

    df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign ,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"FRGMRC", "Type":morphs.to_list()})

    df = df.sort_values(by=['RAh','RAm','RAs'])
    Table.from_pandas(df).write(OUTPUT_PATH+"FRGMRC.fits", format = 'fits', overwrite=True)


# #* Total Sources = 1982
# def create_SGBW22():
#     ra = [[],[],[]]
#     dec = [[],[],[]]
#     types = []
#     sign = []

#     for step in ["test", 'train', 'valid']:
#         for typ in ["Bent", "Compact", "FRI", "FRII"]:
#             for source in os.listdir(f"doc/catalogs/SGBW22/data/{step}/{typ}"):
#                 J2000 = source.rsplit('.', 1)[0]
#                 ra[0].append(J2000[1:3])
#                 ra[1].append(J2000[3:5])
#                 ra[2].append(J2000[5:10])
#                 sign.append(J2000[10])
#                 dec[0].append(J2000[11:13])
#                 dec[1].append(J2000[13:15])
#                 dec[2].append(J2000[15:])

#                 types.append(typ.upper())

#     df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign ,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"SGBW22", "Type":types})

#     df = df.sort_values(by=['RAh','RAm','RAs'])
#     Table.from_pandas(df).write(OUTPUT_PATH+"SGBW22.fits", format = 'fits', overwrite=True)


#! Total Sources = CHECK
def create_Proctor():
    def extractProctor(p1f, label):
        ra = [Angle('{}h{}m{}s'.format(p1f['RAh'][i], p1f['RAm'][i], p1f['RAs'][i])) for i in range(len(p1f['RAh']))]
        dec = [Angle('{}d{}m{}s'.format((2*int(p1f['DE-'][i]=="+")-1)*p1f['DEd'][i], p1f['DEm'][i], p1f['DEs'][i])) for i in range(len(p1f['DE-']))]
        ra_deg = [ra[i].degree for i in range(len(ra))]
        dec_deg = [dec[i].degree for i in range(len(dec))]
        return np.stack((ra_deg ,dec_deg, [label]*len(p1f)), axis=-1).T
    
    # NAT, WAT
    p1f = fits.open("doc/catalogs/Proctor/J_ApJS_194_31_table1.dat.fits")[1].data
    p1 = extractProctor(p1f, "BENT")
    # W-shaped
    p2f = fits.open("doc/catalogs/Proctor/J_ApJS_194_31_table2.dat.fits")[1].data
    p2 = extractProctor(p2f, "OTHER")
    # TB
    p3f = fits.open("doc/catalogs/Proctor/J_ApJS_194_31_table3.dat.fits")[1].data
    p3 = extractProctor(p3f, "BENT")
    # B
    p4f = fits.open("doc/catalogs/Proctor/J_ApJS_194_31_table4.dat.fits")[1].data
    p4 = extractProctor(p4f, "BENT")
    # Ring
    p5f = fits.open("doc/catalogs/Proctor/J_ApJS_194_31_table5.dat.fits")[1].data
    p5 = extractProctor(p5f, "RING")
    #! Ring-like lobe = ?
    p6f = fits.open("doc/catalogs/Proctor/J_ApJS_194_31_table6.dat.fits")[1].data
    p6 = extractProctor(p6f, "OTHER")
    # HYMOR
    p7f = fits.open("doc/catalogs/Proctor/J_ApJS_194_31_table7.dat.fits")[1].data
    p7 = extractProctor(p7f, "OTHER")
    # X-shaped
    p8f = fits.open("doc/catalogs/Proctor/J_ApJS_194_31_table8.dat.fits")[1].data
    p8 = extractProctor(p8f, "X")
    # double-double
    p9f = fits.open("doc/catalogs/Proctor/J_ApJS_194_31_table9.dat.fits")[1].data
    p9 = extractProctor(p9f, "OTHER")
    # core-jet
    p10f = fits.open("doc/catalogs/Proctor/J_ApJS_194_31_table10.dat.fits")[1].data
    p10 = extractProctor(p10f, "OTHER")
    # S/Z shaped
    p11f = fits.open("doc/catalogs/Proctor/J_ApJS_194_31_table11.dat.fits")[1].data
    p11 = extractProctor(p11f, "OTHER")
    #! p12 - Giant radio sources which are not relivant in the study (Out of scope) 
    #! Tri-axial = ?
    p13f = fits.open("doc/catalogs/Proctor/J_ApJS_194_31_table13.dat.fits")[1].data
    p13 = extractProctor(p13f, "OTHER")
    #! Quad Type = ?
    p14f = fits.open("doc/catalogs/Proctor/J_ApJS_194_31_table14.dat.fits")[1].data
    p14 = extractProctor(p14f, "OTHER")
    # other
    p15f = fits.open("doc/catalogs/Proctor/J_ApJS_194_31_table15.dat.fits")[1].data
    p15 = extractProctor(p15f, "OTHER")

    p = np.concatenate((p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p13,p14,p15), axis=1)
    ra= Angle(p[0], unit="deg").hms
    dec = Angle(p[1], unit="deg").dms
    sign = ["-" if "-" in str(p[1,i]) else "+" for i in range(len(p[1]))]
    df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"Proctor", "Type":p[2]})
    for i in range(len(df)):
        df.at[i,"RAh"] = int(df["RAh"][i])
        df.at[i,"RAm"] = int(df["RAm"][i])
        df.at[i,"DECd"] = np.abs(int(df["DECd"][i]))
        df.at[i,"DECm"] = np.abs(int(df["DECm"][i]))
        df.at[i,"DECs"] = np.abs(df["DECs"][i])

    df = df.sort_values(by=['RAh','RAm','RAs'])
    Table.from_pandas(df).write(OUTPUT_PATH+"Proctor.fits", format = 'fits', overwrite=True)


#* Total Sources = 1329
def create_MiraBest():
    mb = np.genfromtxt("doc/catalogs/MiraBest/table1.txt")
    ra = Angle(mb[:,3], unit="hourangle").hms
    dec = Angle(mb[:,4], unit="deg").dms
    sign = ["-" if "-" in str(mb[i,4]) else "+" for i in range(len(mb[:,4]))]
    morph = []

    for encoded_type in mb[:,-1]:
        if encoded_type - 300 >= 0:
            morph.append("OTHER")
        elif encoded_type - 200 >= 0:
            morph.append("FRII")
        elif encoded_type - 100 >= 0:
            morph.append("FRI")
        else:
            print("Error: Unknown type")
            return

    df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"MiraBest", "Type":morph})

    for i in range(len(df)):
        df.at[i,"RAh"] = int(df["RAh"][i])
        df.at[i,"RAm"] = int(df["RAm"][i])
        df.at[i,"DECd"] = np.abs(int(df["DECd"][i]))
        df.at[i,"DECm"] = np.abs(int(df["DECm"][i]))
        df.at[i,"DECs"] = np.abs(df["DECs"][i])

    df = df.sort_values(by=['RAh','RAm','RAs'])
    Table.from_pandas(df).write(OUTPUT_PATH+"MiraBest.fits", format = 'fits', overwrite=True)


#* Total Sources = 1442
def create_LRG():
    lrgFile = open("doc/catalogs/LRG/LRG.txt", "r")
    lrgLines = lrgFile.readlines()
    lrgFile.close()
    lrg = np.empty((2, 1442))
    lrgLabels = np.array([])
    i = 0
    for line in lrgLines:
        if line[0] != 'J':
            continue
        entry = line.split()
        label = ""
        count = 0
        for k, spot in enumerate(entry):
            if "." in spot:
                continue
            count += 1
            if count < 3:
                continue
            label = spot
            break
            
            
        lrg[0][i] = Angle(entry[1], unit='hour').degree
        lrg[1][i] = entry[2]
        if label == '1':
            lrgLabels = np.append(lrgLabels, 'COMPACT')
        elif label == '2':
            lrgLabels = np.append(lrgLabels, 'FRI')
        elif label == '3':
            lrgLabels = np.append(lrgLabels, 'FRII')
        elif label == '4':
            lrgLabels = np.append(lrgLabels, 'BENT')
        elif label == '5':
            lrgLabels = np.append(lrgLabels, 'X')
        elif label == '6':
            lrgLabels = np.append(lrgLabels, 'RING')
        else:
            lrgLabels = np.append(lrgLabels, 'OTHER')
        i += 1
            
    lrg = np.append(lrg, [lrgLabels], axis=0)

    ra= Angle(lrg[0], unit="deg").hms
    dec = Angle(lrg[1], unit="deg").dms
    sign = ["-" if "-" in str(lrg[1,i]) else "+" for i in range(len(lrg[1]))]
    df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"LRG", "Type":lrg[2]})
    for i in range(len(df)):
        df.at[i,"RAh"] = int(df["RAh"][i])
        df.at[i,"RAm"] = int(df["RAm"][i])
        df.at[i,"DECd"] = np.abs(int(df["DECd"][i]))
        df.at[i,"DECm"] = np.abs(int(df["DECm"][i]))
        df.at[i,"DECs"] = np.abs(df["DECs"][i])
    df = df.sort_values(by=['RAh','RAm','RAs'])
    Table.from_pandas(df).write(OUTPUT_PATH+"LRG.fits", format = 'fits', overwrite=True)


#* Total Sources = 2158
def create_GKCBR23():
    ra = [[],[],[]]
    dec = [[],[],[]]
    types = []
    sign = []

    for typ in ["Bent", "Compact", "FRI", "FRII"]:
        for source in os.listdir(f"doc/catalogs/GKCBR23/galaxy_data/all/{typ}"):
            
            angle = Angle(source.split('_')[0], unit='degree').hms
            ra[0].append(np.abs(angle[0]))
            ra[1].append(np.abs(angle[1]))
            ra[2].append(np.abs(angle[2]))
            angle = Angle(source.split('_')[1], unit='degree').dms
            dec[0].append(np.abs(angle[0]))
            dec[1].append(np.abs(angle[1]))
            dec[2].append(np.abs(angle[2]))
            sign.append("+" if float(source.split('_')[1]) >= 0 else "-")
            types.append(typ.upper())

    df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign ,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"GKCBR23", "Type":types})

    df = df.sort_values(by=['RAh','RAm','RAs'])
    Table.from_pandas(df).write(OUTPUT_PATH+"GKCBR23.fits", format = 'fits', overwrite=True)

#* Total Sources = 123
def create_FRIICAT():
    file = fits.open("doc/catalogs/FRIICAT/table1.dat.fits")[1].data
    coords = np.array([[],[]])
    names = file["SDSS"].tolist()
    for i in range(len(names)):
        ra = Angle('{}h{}m{}s'.format(names[i][1:3],names[i][3:5],names[i][5:10])).degree
        dec = Angle('{}d{}m{}s'.format(names[i][10:13],names[i][13:15],names[i][15:19])).degree 
        coords = np.append(coords,np.array([[ra],[dec]]), axis=1)
    ra= Angle(coords[0], unit="deg").hms
    dec = Angle(coords[1], unit="deg").dms
    sign = ["-" if "-" in str(coords[1,i]) else "+" for i in range(len(coords[1]))]
    df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"FRIICAT", "Type":"FRII"})
    for i in range(len(df)):
        df.at[i,"RAh"] = int(df["RAh"][i])
        df.at[i,"RAm"] = int(df["RAm"][i])
        df.at[i,"DECd"] = np.abs(int(df["DECd"][i]))
        df.at[i,"DECm"] = np.abs(int(df["DECm"][i]))
        df.at[i,"DECs"] = np.abs(df["DECs"][i])
    df = df.sort_values(by=['RAh','RAm','RAs'])
    Table.from_pandas(df).write(OUTPUT_PATH+"FRIICAT.fits", format = 'fits', overwrite=True)

#* Total Sources = 233
def create_FRICAT():
    coords = np.array([[],[]])
    for table in ["1", "2"]:
        file = fits.open(f"doc/catalogs/FRICAT/tableb{table}.dat.fits")[1].data
        names = file["SDSS"].tolist()

        for i in range(len(names)):
            ra = Angle('{}h{}m{}s'.format(names[i][1:3],names[i][3:5],names[i][5:10])).degree
            dec = Angle('{}d{}m{}s'.format(names[i][10:13],names[i][13:15],names[i][15:19])).degree 
            coords = np.append(coords,np.array([[ra],[dec]]), axis=1)
    
    ra= Angle(coords[0], unit="deg").hms
    dec = Angle(coords[1], unit="deg").dms
    sign = ["-" if "-" in str(coords[1,i]) else "+" for i in range(len(coords[1]))]
    df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"FRICAT", "Type":"FRI"})
    for i in range(len(df)):
        df.at[i,"RAh"] = int(df["RAh"][i])
        df.at[i,"RAm"] = int(df["RAm"][i])
        df.at[i,"DECd"] = np.abs(int(df["DECd"][i]))
        df.at[i,"DECm"] = np.abs(int(df["DECm"][i]))
        df.at[i,"DECs"] = np.abs(df["DECs"][i])
    df = df.sort_values(by=['RAh','RAm','RAs'])
    Table.from_pandas(df).write(OUTPUT_PATH+"FRICAT.fits", format = 'fits', overwrite=True)


#* Total Sources = 108
def create_FR0CAT():
    file = fits.open("doc/catalogs/FR0CAT/tablea1.dat.fits")[1].data
    coords = np.array([[],[]])
    names = file["SDSS"].tolist()
    for i in range(len(names)):
        ra = Angle('{}h{}m{}s'.format(names[i][1:3],names[i][3:5],names[i][5:10])).degree
        dec = Angle('{}d{}m{}s'.format(names[i][10:13],names[i][13:15],names[i][15:19])).degree 
        coords = np.append(coords,np.array([[ra],[dec]]), axis=1)
    ra= Angle(coords[0], unit="deg").hms
    dec = Angle(coords[1], unit="deg").dms
    sign = ["-" if "-" in str(coords[1,i]) else "+" for i in range(len(coords[1]))]
    df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"FR0CAT", "Type":"COMPACT"})
    for i in range(len(df)):
        df.at[i,"RAh"] = int(df["RAh"][i])
        df.at[i,"RAm"] = int(df["RAm"][i])
        df.at[i,"DECd"] = np.abs(int(df["DECd"][i]))
        df.at[i,"DECm"] = np.abs(int(df["DECm"][i]))
        df.at[i,"DECs"] = np.abs(df["DECs"][i])
    df = df.sort_values(by=['RAh','RAm','RAs'])
    Table.from_pandas(df).write(OUTPUT_PATH+"FR0CAT.fits", format = 'fits', overwrite=True)


#* Total Sources = 2100
def create_CRUMB():
    df = pd.read_csv("doc/catalogs/CRUMB/catalogue.txt")
    ra = Angle(df["Ra"], unit="hourangle").hms
    dec = Angle(df["Dec"], unit="degree").dms
    sign = ["-" if "-" in str(df["Dec"][i]) else "+" for i in range(len(df["Dec"]))]

    df.replace("Hyb","OTHER", inplace=True)

    df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"CRUMB", "Type":df["Type"]})
    for i in range(len(df)):
        df.at[i,"RAh"] = int(df["RAh"][i])
        df.at[i,"RAm"] = int(df["RAm"][i])
        df.at[i,"DECd"] = np.abs(int(df["DECd"][i]))
        df.at[i,"DECm"] = np.abs(int(df["DECm"][i]))
        df.at[i,"DECs"] = np.abs(df["DECs"][i])
    df = df.sort_values(by=['RAh','RAm','RAs'])
    Table.from_pandas(df).write(OUTPUT_PATH+"CRUMB.fits", format = 'fits', overwrite=True)


#! Total Sources = 858 (857+1 duplicate) (Paper says 859?? Toothless also says 859)
def create_CoNFIG():
    ra = [[],[],[]]
    dec = [[],[],[]]
    sign = []
    typ = []
    
    for sample in ["1", "2", "3", "4"]:    
        with open(f"doc/catalogs/CoNFIG/config{sample}.dat.txt", "r") as f:
            for line in f.readlines():

                ra[0].append(int(line[4:6]))
                ra[1].append(int(line[7:9]))
                ra[2].append(float(line[10:15]))
                sign.append(str(line[16]))
                dec[0].append(int(line[17:19]))
                dec[1].append(int(line[20:22]))
                dec[2].append(float(line[23:28]))

                encoded_type = line[58:60].strip()
                if encoded_type in ["C", "C*", "S*"]:
                    typ.append("COMPACT")
                elif encoded_type == "I":
                    typ.append("FRI")
                elif encoded_type == "II":
                    typ.append("FRII")
                elif encoded_type == "U":
                    typ.append("OTHER")
                else:
                    print("Error: Unknown type")
                    return
            
        df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"CoNFIG", "Type":typ})
        for i in range(len(df)):
            df.at[i,"RAh"] = int(df["RAh"][i])
            df.at[i,"RAm"] = int(df["RAm"][i])
            df.at[i,"DECd"] = np.abs(int(df["DECd"][i]))
            df.at[i,"DECm"] = np.abs(int(df["DECm"][i]))
            df.at[i,"DECs"] = np.abs(float(df["DECs"][i]))
        df = df.sort_values(by=['RAh','RAm','RAs'])
        Table.from_pandas(df).write(OUTPUT_PATH+"CoNFIG.fits", format = 'fits', overwrite=True)


#* Total Sources = 658
def create_FRDEEP():
    df_c = pd.read_csv("doc/catalogs/FR-DEEP/CoNFIG_II_full_Table.csv")
    df_f = pd.read_csv("doc/catalogs/FR-DEEP/FRICAT_Table1.csv")

    ra = [[],[],[]]
    dec = [[],[],[]]
    sign = []
    typ = []

    for file in os.listdir(f"doc/catalogs/FR-DEEP/FIRST"):
        if file.endswith(".fits"):
            name, morph = file.rsplit(".", 1)[0].split("_")

            if morph == "I":
                typ.append("FRI")
            elif morph == "II":
                typ.append("FRII")
            else:
                raise ValueError("Unknown type", morph)

            if name.startswith("SDSS"):
                entry = df_f[df_f["SimbadName"] == name][["_RAJ2000", "_DEJ2000"]]
            else:
                entry = df_c[df_c["Name"] == name][["_RAJ2000", "_DEJ2000"]]

            h,m,s = entry["_RAJ2000"].values[0].split(" ")
            ra[0].append(int(h))
            ra[1].append(int(m))
            ra[2].append(float(s))
            d,m,s = entry["_DEJ2000"].values[0].split(" ")
            dec[0].append(int(d))
            dec[1].append(int(m))
            dec[2].append(float(s))
            sign.append(d[0])

    df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"FRDEEP", "Type":typ})
    for i in range(len(df)):
        df.at[i,"RAh"] = int(df["RAh"][i])
        df.at[i,"RAm"] = int(df["RAm"][i])
        df.at[i,"DECd"] = np.abs(int(df["DECd"][i]))
        df.at[i,"DECm"] = np.abs(int(df["DECm"][i]))
        df.at[i,"DECs"] = np.abs(float(df["DECs"][i]))
    df = df.sort_values(by=['RAh','RAm','RAs'])
    Table.from_pandas(df).write(OUTPUT_PATH+"FRDEEP.fits", format = 'fits', overwrite=True)
        
def correct_AT17_error(original_name, simbad=False):
    if simbad:
        print("Finding:", original_name)
        result = Simbad.query_object(original_name)
        if result is None:
            print("Error: Simbad could not find", original_name,"-> Dropping...")
            return None
         
        # print(result[["RA","DEC"]])
        rightAscension = result["RA"].data[0].split(" ")
        declination = result["DEC"].data[0].split(" ")
        if len(declination) == 2:
            declination = [declination[0], "00", declination[1]]
        return rightAscension, declination
    
    elif original_name.startswith("J"):
        if "+" not in original_name and "-" not in original_name:
            return original_name[:10] + "+" + original_name[10:]
        return original_name

    elif original_name.startswith("TXS") and original_name[7].isdigit():
        return original_name[:7] + '-' + original_name[7:]

    else:
        return {
            'EQSB14070231' : 'EQSB1407-0231',
            'MRC1408030' : 'MRC1408-030',
        }.get(original_name, original_name)
    
# #! Total Sources = CHECK
def create_AT17():
    df_c = pd.read_csv("doc/catalogs/FR-DEEP/CoNFIG_II_full_Table.csv")[["_RAJ2000", "_DEJ2000", "Name", "SimbadName"]]
    df_c["Name"] = df_c["Name"].str.replace(" ", "")
    df_c["SimbadName"] = df_c["SimbadName"].str.replace(" ", "")
    
    # Obtaining sources
    initial = []
    ra = [[],[],[]]
    dec = [[],[],[]]
    sign = []
    morph = []

    for file, typ in zip(["bent-list.txt", "fr1-full.txt", "fr2-full.txt"], ["BENT", "FRI", "FRII"]):
        
        with open(f"doc/catalogs/AT17-Toothless/{file}", "r") as f:
            extracted = [x.strip() for x in f.readlines() if x.strip() != ""]
            initial.extend(extracted)

        corrected = set([correct_AT17_error(x) for x in extracted])

        if file != "fr2-full.txt":
            J2000 = set([x for x in corrected if x.startswith("J")])
        else:
            J2000 = set()
        ex_J2000 = corrected - J2000

        # First dealing with the non-J2000 sources --------------------------------
        match_on_name = df_c[df_c["Name"].isin(ex_J2000)]
        ex_J2000 = ex_J2000 - set(match_on_name["Name"])
        match_on_simbadName = df_c[df_c["SimbadName"].isin(ex_J2000)]
        ex_J2000 = ex_J2000 - set(match_on_simbadName["SimbadName"])

        df = pd.concat([match_on_name, match_on_simbadName])[["_RAJ2000", "_DEJ2000"]]

        matched_simbad = []
        failed = []

        for x in ex_J2000:
            coord = correct_AT17_error(x, True)
            if coord is not None:
                matched_simbad.append(coord)
            else:
                failed.append(x)

        for _, entry in df.iterrows():
            h,m,s = entry["_RAJ2000"].split(" ")
            ra[0].append(int(h))
            ra[1].append(int(m))
            ra[2].append(float(s))
            d,m,s = entry["_DEJ2000"].split(" ")
            dec[0].append(int(d))
            dec[1].append(int(m))
            dec[2].append(abs(float(s)))
            sign.append(d[0])
            morph.append(typ)

        for entry in matched_simbad:
            h,m,s = entry[0]
            ra[0].append(int(h))
            ra[1].append(int(m))
            ra[2].append(float(s))
            d,m,s = entry[1]
            dec[0].append(int(d))
            dec[1].append(int(m))
            dec[2].append(abs(float(s)))
            sign.append(d[0])
            morph.append(typ)

        # Now dealing with the J2000 sources --------------------------------
        for source in J2000:
            seperate = "-" if "-" in source else "+"
            Ra, Dec = source.split(seperate)
            ra[0].append(int(Ra[1:3]))
            ra[1].append(int(Ra[3:5]))
            ra[2].append(float(Ra[5:]))
            dec[0].append(int(Dec[:2]))
            dec[1].append(int(Dec[2:4]))
            dec[2].append(float(Dec[4:]))
            sign.append(seperate)
            morph.append(typ)

        print("<<New File>>", file)
        print("-J20000:",len(J2000))
        print("-Matched on Name:", len(match_on_name))
        print("-Matched on SimbadName:", len(match_on_simbadName))
        print("-Matched on Simbad:", len(matched_simbad))
        print("-Failed:", len(failed))
        print("--------------------")

    print("Initial, Final:", len(initial), len(ra[0]),"diff=", len(initial)-len(ra[0]))
    print("--------------------")

    df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"AT17", "Type":morph})
    for i in range(len(df)):
        df.at[i,"RAh"] = int(df["RAh"][i])
        df.at[i,"RAm"] = int(df["RAm"][i])
        df.at[i,"DECd"] = np.abs(int(df["DECd"][i]))
        df.at[i,"DECm"] = np.abs(int(df["DECm"][i]))
        df.at[i,"DECs"] = np.abs(float(df["DECs"][i]))
    df = df.sort_values(by=['RAh','RAm','RAs'])
    Table.from_pandas(df).write(OUTPUT_PATH+"AT17.fits", format = 'fits', overwrite=True)