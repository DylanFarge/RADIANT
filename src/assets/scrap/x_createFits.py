import os
import numpy as np
import pandas as pd
from astropy.coordinates import Angle
import astropy.units as u
import astropy.io.fits as fits
from astropy.table import Table

def readAndSaveMB(write = True):
    miraBestTable1 = fits.open("catalogs/originals/MiraBest/table1.dat.fits")[1].data
    mb1 = np.stack((Angle(miraBestTable1['RAhours'], unit='hour').degree , miraBestTable1['DEdeg'], ["OTHER"]*len(miraBestTable1)), axis=-1).T

    miraBestTable4 = fits.open("catalogs/originals/MiraBest/table4.dat.fits")[1].data
    mb4 = np.empty((2, len(miraBestTable4)))
    mb4Labels = np.array([])
    for i, source in enumerate(miraBestTable4):
        mb4[0][i] = Angle(source['RAhours'], unit='hour').degree
        mb4[1][i] = source['DEdeg']
        if source["c/e"] == 0:
            mb4Labels = np.append(mb4Labels,"COMPACT")
        elif source["c/e"] == 1:
            mb4Labels = np.append(mb4Labels,"FRI")
        elif source["c/e"] == 2:
            mb4Labels = np.append(mb4Labels,"FRII")
        else:
            mb4Labels = np.append(mb4Labels,"OTHER")
            
    mb = np.append(mb4, [mb4Labels], axis=0)
    mb = np.append(mb, mb1, axis=1)

    ra= Angle(mb[0], unit="deg").hms
    dec = Angle(mb[1], unit="deg").dms
    print("--------------------")
    print(mb[1])
    sign = ["-" if "-" in str(mb[1,i]) else "+" for i in range(len(mb[1]))]
    df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign ,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"MiraBest", "Type":mb[2]})
    for i in range(len(df)):
        df["RAh"][i] = int(df["RAh"][i])
        df["RAm"][i] = int(df["RAm"][i])
        df["DECd"][i] = np.abs(int(df["DECd"][i]))
        df["DECm"][i] = np.abs(int(df["DECm"][i]))
        df["DECs"][i] = np.abs(df["DECs"][i])
    df = df.sort_values(by=['RAh','RAm','RAs'])
    t = Table.from_pandas(df)
    if write:
        t.write("MiraBest.fits", format = 'fits')
    
def readAndSaveLRG(write = True):
    lrgFile = open("catalogs/originals/LRG/LRG.txt", "r")
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
        df["RAh"][i] = int(df["RAh"][i])
        df["RAm"][i] = int(df["RAm"][i])
        df["DECd"][i] = np.abs(int(df["DECd"][i]))
        df["DECm"][i] = np.abs(int(df["DECm"][i]))
        df["DECs"][i] = np.abs(df["DECs"][i])
    df = df.sort_values(by=['RAh','RAm','RAs'])
    t = Table.from_pandas(df)
    if write:
      t.write("LRG.fits", format = 'fits')
    
def readAndSaveUNLRG(write = True):
    unlrgFile = open("catalogs/originals/unLRG/unLRG.txt", "r")
    unlrgLines = unlrgFile.readlines()
    unlrgFile.close()
    unlrg = np.empty((2, 14245))
    unlrgLabels = np.array([])
    i = 0
    for line in unlrgLines:
        if line[0] != 'J':
            continue
        entry = line.split()
        unlrg[0][i] = Angle(entry[1], unit='hour').degree
        unlrg[1][i] = entry[2]
        print(entry[6])
        if entry[6] == '1' or entry[6] == '1F':
            unlrgLabels = np.append(unlrgLabels, 'COMPACT')
        elif entry[6] == '2':
            unlrgLabels = np.append(unlrgLabels, 'FRI')
        elif entry[6] == '3':
            unlrgLabels = np.append(unlrgLabels, 'FRII')
        elif entry[6] == '4':
            unlrgLabels = np.append(unlrgLabels, 'BENT')
        elif entry[6] == '5':
            unlrgLabels = np.append(unlrgLabels, 'X')
        elif entry[6] == '6':
            unlrgLabels = np.append(unlrgLabels, 'RING')
        else:
            unlrgLabels = np.append(unlrgLabels, 'OTHER')
        i += 1
            
    unlrg = np.append(unlrg, [unlrgLabels], axis=0)

    ra= Angle(unlrg[0], unit="deg").hms
    dec = Angle(unlrg[1], unit="deg").dms
    sign = ["-" if "-" in str(unlrg[1,i]) else "+" for i in range(len(unlrg[1]))]
    df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"unLRG", "Type":unlrg[2]})
    for i in range(len(df)):
        df["RAh"][i] = int(df["RAh"][i])
        df["RAm"][i] = int(df["RAm"][i])
        df["DECd"][i] = np.abs(int(df["DECd"][i]))
        df["DECm"][i] = np.abs(int(df["DECm"][i]))
        df["DECs"][i] = np.abs(df["DECs"][i])
    df = df.sort_values(by=['RAh','RAm','RAs'])
    t = Table.from_pandas(df)
    if write:
        t.write("unLRG.fits", format = 'fits')

def readAndSavePROCTOR(write = True):
    def extractProctor(p1f, label):
        ra = [Angle('{}h{}m{}s'.format(p1f['RAh'][i], p1f['RAm'][i], p1f['RAs'][i])) for i in range(len(p1f['RAh']))]
        dec = [Angle('{}d{}m{}s'.format((2*int(p1f['DE-'][i]=="+")-1)*p1f['DEd'][i], p1f['DEm'][i], p1f['DEs'][i])) for i in range(len(p1f['DE-']))]
        ra_deg = [ra[i].degree for i in range(len(ra))]
        dec_deg = [dec[i].degree for i in range(len(dec))]
        return np.stack((ra_deg ,dec_deg, [label]*len(p1f)), axis=-1).T
    #other
    #! NAT,WAT = BENT
    p1f = fits.open("catalogs/originals/Proctor/J_ApJS_194_31_table1.dat.fits")[1].data
    p1 = extractProctor(p1f, "OTHER")
    #other
    p2f = fits.open("catalogs/originals/Proctor/J_ApJS_194_31_table2.dat.fits")[1].data
    p2 = extractProctor(p2f, "OTHER")
    #bent
    p3f = fits.open("catalogs/originals/Proctor/J_ApJS_194_31_table3.dat.fits")[1].data
    p3 = extractProctor(p3f, "BENT")
    #bent
    p4f = fits.open("catalogs/originals/Proctor/J_ApJS_194_31_table4.dat.fits")[1].data
    p4 = extractProctor(p4f, "BENT")
    #ring
    p5f = fits.open("catalogs/originals/Proctor/J_ApJS_194_31_table5.dat.fits")[1].data
    p5 = extractProctor(p5f, "RING")
    #other
    #! RING? Need to go look at the images and the article
    p6f = fits.open("catalogs/originals/Proctor/J_ApJS_194_31_table6.dat.fits")[1].data
    p6 = extractProctor(p6f, "OTHER")
    #other
    p7f = fits.open("catalogs/originals/Proctor/J_ApJS_194_31_table7.dat.fits")[1].data
    p7 = extractProctor(p7f, "OTHER")
    # X
    p8f = fits.open("catalogs/originals/Proctor/J_ApJS_194_31_table8.dat.fits")[1].data
    p8 = extractProctor(p8f, "X")
    #other
    p9f = fits.open("catalogs/originals/Proctor/J_ApJS_194_31_table9.dat.fits")[1].data
    p9 = extractProctor(p9f, "OTHER")
    p10f = fits.open("catalogs/originals/Proctor/J_ApJS_194_31_table10.dat.fits")[1].data
    p10 = extractProctor(p10f, "OTHER")
    p11f = fits.open("catalogs/originals/Proctor/J_ApJS_194_31_table11.dat.fits")[1].data
    p11 = extractProctor(p11f, "OTHER")
    #! p12 - Giant radio sources which are not relivant in the study (Out of scope) 
    p13f = fits.open("catalogs/originals/Proctor/J_ApJS_194_31_table13.dat.fits")[1].data
    p13 = extractProctor(p13f, "OTHER") #! What is this? Look at images
    p14f = fits.open("catalogs/originals/Proctor/J_ApJS_194_31_table14.dat.fits")[1].data
    p14 = extractProctor(p14f, "OTHER") #! What is this? Look at images
    p15f = fits.open("catalogs/originals/Proctor/J_ApJS_194_31_table15.dat.fits")[1].data
    p15 = extractProctor(p15f, "OTHER") #! What is this? Look at images

    p = np.concatenate((p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p13,p14,p15), axis=1)
    print(p.shape)

    ra= Angle(p[0], unit="deg").hms
    dec = Angle(p[1], unit="deg").dms
    sign = ["-" if "-" in str(p[1,i]) else "+" for i in range(len(p[1]))]
    df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"Proctor", "Type":p[2]})
    for i in range(len(df)):
        df["RAh"][i] = int(df["RAh"][i])
        df["RAm"][i] = int(df["RAm"][i])
        df["DECd"][i] = np.abs(int(df["DECd"][i]))
        df["DECm"][i] = np.abs(int(df["DECm"][i]))
        df["DECs"][i] = np.abs(df["DECs"][i])
    df = df.sort_values(by=['RAh','RAm','RAs'])
    t = Table.from_pandas(df)
    if write:
        t.write("Proctor.fits", format = 'fits')
    
def readAndSaveFR0(write = True):
    FR0 = fits.open("catalogs/originals/FR0CAT/tablea1.dat.fits")[1].data
    FR0_coords = np.array([[],[]])
    FR_names = FR0["SDSS"].tolist()
    for i in range(len(FR_names)):
        ra = Angle('{}h{}m{}s'.format(FR_names[i][1:3],FR_names[i][3:5],FR_names[i][5:10])).degree
        dec = Angle('{}d{}m{}s'.format(FR_names[i][10:13],FR_names[i][13:15],FR_names[i][15:19])).degree 
        FR0_coords = np.append(FR0_coords,np.array([[ra],[dec]]), axis=1)
        
    ra= Angle(FR0_coords[0], unit="deg").hms
    dec = Angle(FR0_coords[1], unit="deg").dms
    # print(FR0_coords[1,0])
    sign = ["-" if "-" in str(FR0_coords[1,i]) else "+" for i in range(len(FR0_coords[1]))]
    df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"FR0", "Type":"COMPACT"})
    for i in range(len(df)):
        df["RAh"][i] = int(df["RAh"][i])
        df["RAm"][i] = int(df["RAm"][i])
        df["DECd"][i] = np.abs(int(df["DECd"][i]))
        df["DECm"][i] = np.abs(int(df["DECm"][i]))
        df["DECs"][i] = np.abs(df["DECs"][i])
    df = df.sort_values(by=['RAh','RAm','RAs'])
    t = Table.from_pandas(df)
    if write:
        t.write("FR0.fits", format = 'fits')
    
def readAndSaveFRI(write = True):
    FRI = fits.open("catalogs/originals/FRICAT/tableb1.dat.fits")[1].data
    FRI_coords = np.array([[],[]])
    FR_names = FRI["SDSS"].tolist()
    for i in range(len(FR_names)):
        ra = Angle('{}h{}m{}s'.format(FR_names[i][1:3],FR_names[i][3:5],FR_names[i][5:10])).degree
        dec = Angle('{}d{}m{}s'.format(FR_names[i][10:13],FR_names[i][13:15],FR_names[i][15:19])).degree 
        FRI_coords = np.append(FRI_coords,np.array([[ra],[dec]]), axis=1)
    FRI = fits.open("catalogs/originals/FRICAT/tableb2.dat.fits")[1].data
    FR_names = FRI["SDSS"].tolist()
    for i in range(len(FR_names)):
        ra = Angle('{}h{}m{}s'.format(FR_names[i][1:3],FR_names[i][3:5],FR_names[i][5:10])).degree
        dec = Angle('{}d{}m{}s'.format(FR_names[i][10:13],FR_names[i][13:15],FR_names[i][15:19])).degree 
        FRI_coords = np.append(FRI_coords,np.array([[ra],[dec]]), axis=1)
        
    ra= Angle(FRI_coords[0], unit="deg").hms
    dec = Angle(FRI_coords[1], unit="deg").dms
    sign = ["-" if "-" in str(FRI_coords[1,i]) else "+" for i in range(len(FRI_coords[1]))]
    df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"FRI", "Type":"FRI"})
    for i in range(len(df)):
        df["RAh"][i] = int(df["RAh"][i])
        df["RAm"][i] = int(df["RAm"][i])
        df["DECd"][i] = np.abs(int(df["DECd"][i]))
        df["DECm"][i] = np.abs(int(df["DECm"][i]))
        df["DECs"][i] = np.abs(df["DECs"][i])
    df = df.sort_values(by=['RAh','RAm','RAs'])
    t = Table.from_pandas(df)
    if write:
        t.write("FRI.fits", format = 'fits')
    
def readAndSaveFRII(write = True):
    FRII = fits.open("catalogs/originals/FRIICAT/table1.dat.fits")[1].data
    FRII_coords = np.array([[],[]])
    FR_names = FRII["SDSS"].tolist()
    for i in range(len(FR_names)):
        ra = Angle('{}h{}m{}s'.format(FR_names[i][1:3],FR_names[i][3:5],FR_names[i][5:10])).degree
        dec = Angle('{}d{}m{}s'.format(FR_names[i][10:13],FR_names[i][13:15],FR_names[i][15:19])).degree 
        FRII_coords = np.append(FRII_coords,np.array([[ra],[dec]]), axis=1)
    ra= Angle(FRII_coords[0], unit="deg").hms
    dec = Angle(FRII_coords[1], unit="deg").dms
    sign = ["-" if "-" in str(FRII_coords[1,i]) else "+" for i in range(len(FRII_coords[1]))]
    df = pd.DataFrame({"RAh":ra[0],"RAm":ra[1],"RAs":ra[2], "DECsign": sign,"DECd":dec[0],"DECm":dec[1],"DECs":dec[2],"Catalog":"FRII", "Type":"FRII"})
    for i in range(len(df)):
        df["RAh"][i] = int(df["RAh"][i])
        df["RAm"][i] = int(df["RAm"][i])
        df["DECd"][i] = np.abs(int(df["DECd"][i]))
        df["DECm"][i] = np.abs(int(df["DECm"][i]))
        df["DECs"][i] = np.abs(df["DECs"][i])
    df = df.sort_values(by=['RAh','RAm','RAs'])
    t = Table.from_pandas(df)
    if write:
        t.write("FRII.fits", format = 'fits')
    
print(os.getcwd())
# readAndSaveMB(write =False)
# readAndSaveFR0(write =False)
# readAndSaveFRI(write =False)s
# readAndSaveFRII(write =False)
# readAndSavePROCTOR(write =False)
readAndSaveUNLRG(write =False)
# readAndSaveLRG(write =False)