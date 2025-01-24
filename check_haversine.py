from astropy.coordinates import SkyCoord
import numpy as np
import numexpr as ne

def sphere_dist(ra1, dec1, ra2, dec2):
    ra1 = np.radians(ra1).astype(np.float64)
    ra2 = np.radians(ra2).astype(np.float64)
    dec1 = np.radians(dec1).astype(np.float64)
    dec2 = np.radians(dec2).astype(np.float64)
    numerator = ne.evaluate('sin((dec2 - dec1) / 2) ** 2 + cos(dec1) * cos(dec2) * sin((ra2 - ra1) / 2) ** 2')
    dists = ne.evaluate('2 * arctan2(numerator ** 0.5, (1 - numerator) ** 0.5)')
    return np.degrees(dists)

def haversine(ra1, dec1, ra2, dec2, unit):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])

    # haversine formula
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = np.sin(ddec/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra/2)**2
    dist = 2 * np.arcsin(np.sqrt(a))
    multiplier = 1
    if unit == 'arcsec':
        multiplier = 3600
    return np.degrees(dist) * multiplier

#! Dist = 0.0003704 deg. Same source, different catalogs.
# s1 = SkyCoord(ra=228.849,dec=10.31 ,unit="deg")
# s2 = SkyCoord(ra=228.848916,dec=10.3103611,unit="deg")

#! Dist = 0.07 deg. Different source, different catalogs.
# s1 = SkyCoord(ra=227.80155,dec=10.02232 ,unit="deg")
# s2 = SkyCoord(ra=227.873,dec=10.029,unit="deg")

#! Dist = 0.002 deg. Same source, different catalogs.
# s1 = SkyCoord(ra=	140.291,dec=		12.99 ,unit="deg")
# s2 = SkyCoord(ra=		140.2906185,dec=		12.9899,unit="deg")

#! Dist = 0.008 deg. Same source, different catalogs.
# s1 = SkyCoord(ra=138.0556665,dec=	13.6542028,unit="deg")
# s2 = SkyCoord(ra=138.055,dec=	13.646,unit="deg")

#* Dist = 0.010 deg. Same source, different catalogs.
# s1 = SkyCoord(ra=146.947,dec=	7.421,unit="deg")
# s2 = SkyCoord(ra=146.936667,dec=		7.42075,unit="deg")

# Check out this location:
#Proctor RA:127.1802958333 DEC:24.61745 CoreType:v?

#? First image says to be 0.0324
# s1 = SkyCoord(ra=217.5139166667, dec=7.2503611111, unit="deg")
# s2 = SkyCoord(ra=217.48074, dec=7.25357, unit="deg")

#? Second image says to be 0.049
s1 = SkyCoord(ra=217.514, dec=7.25, unit="deg")
s2 = SkyCoord(ra=217.48074, dec=7.25357, unit="deg")

print(s1.separation(s2).deg)
print(haversine(s1.ra.deg, s1.dec.deg, s2.ra.deg, s2.dec.deg, 'deg'))
print(sphere_dist(s1.ra.deg, s1.dec.deg, s2.ra.deg, s2.dec.deg))