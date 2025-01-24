import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_uvw(xyz, lambdas, dec, h):
    uvw = []
    for X, Y, Z in xyz:
        uvw.append([
            [lam**(-1) * np.tile((np.sin(h) * X + np.cos(h) * Y), dec.shape) for lam in lambdas], #u
            [lam**(-1) * (-np.sin(dec) * np.cos(h) * X + np.sin(dec) * np.sin(h) * Y + np.cos(dec) * Z) for lam in lambdas], #v
            [lam**(-1) * (np.cos(dec) * np.cos(h) * X - np.cos(dec) * np.sin(h) * Y + np.sin(dec) * Z) for lam in lambdas], #w
        ])
    return np.array(uvw)


def lat_from_string(latitude):
    '''Latitude must be in format xxd xx' xx.xx"'''
    latitude = latitude.strip()
    deg, remain = latitude.split("d")
    arcms = remain.split("'")
    deg = int(deg.strip())
    deg_sign = deg/abs(deg)
    arcm = int(arcms[0].strip())
    arcs = float(arcms[1].strip())
    return deg_sign * (abs(deg) + arcm/60 + arcs/3600) * np.pi/180


def get_XYZ(antennae, L):
    xyz, distances = [], []
    for i in range(len(antennae)):
        for j in range(i+1, len(antennae)):

            baseline = antennae[j] - antennae[i]
            dist = np.linalg.norm(baseline)
            azimuth = np.arctan2(baseline[0], baseline[1])
            elevation = np.arcsin(baseline[2]/dist)

            xyz.append([
                dist * (np.cos(L) * np.sin(elevation) - np.sin(L) * np.cos(elevation) * np.cos(azimuth)),#X
                dist * (np.cos(elevation) * np.sin(azimuth)),                                    #Y
                dist * (np.sin(L) * np.sin(elevation) + np.cos(L) * np.cos(elevation) * np.cos(azimuth)) #Z
            ])
            
            distances.append(dist)
    return np.array(xyz), np.array(distances)


def get_field_centre_declinations(sources):
    df = pd.read_csv("ConstructCatalog/results/RADCAT_ML.csv", index_col=0)
    return np.array([np.radians(df.loc[sources, "DEC/deg"])]).T


def plot_fourier(X, X_shifted, X_inv_fourier, X_fourier_shifted, u_offset, v_offset):
    extent = [-u_offset, u_offset, -v_offset, v_offset]
    source = 0

    for kind in ["Absolute", "Real", "Imaginary"]:
        fig, ax = plt.subplots(4,3, figsize=(15,20))
        for i in range(X.shape[-1]):


            ax[0,i].imshow(X[source,:,:,i], origin='lower')


            ax[1,i].imshow(X_shifted[source,:,:,i], origin='lower')
            ax[1,i].axis("off")
            ax[1,i].set_title("Shifted") if i == 0 else None

            if kind == "Absolute":
                ax[2,i].imshow(abs(X_inv_fourier)[source,:,:,i], origin='lower')
                ax[3,i].imshow(abs(X_fourier_shifted)[source,:,:,i], extent=extent, origin='lower')
            elif kind == "Real":
                ax[2,i].imshow(X_inv_fourier[source,:,:,i].real)
                ax[3,i].imshow(X_fourier_shifted[source,:,:,i].real, extent=extent, origin='lower')
            elif kind == "Imaginary":
                ax[2,i].imshow(X_inv_fourier[source,:,:,i].imag)
                ax[3,i].imshow(X_fourier_shifted[source,:,:,i].imag, extent=extent, origin='lower')

            ax[2,i].axis("off")
            ax[2,i].set_title("Inv Fourier") if i == 0 else None

            ax[3,i].set_title("Fourier Shifted") if i == 0 else None

        fig.savefig("ConstructVisibilities/results/fourier_" + kind + ".png")


def plot_uv_tracks(uv):
    print("Plotting UV tracks...",end="")
    # UV: baselines, coords, catalogs, sources, timesteps
    fig, ax = plt.subplots(1,uv.shape[2],figsize=(5*uv.shape[2],5))

    for catalog in range(uv.shape[2]):

        ax[catalog].set_title(f"{catalog}")
        ax[catalog].set_xlabel("U")
        ax[catalog].set_ylabel("V")
        ax[catalog].set_aspect('equal', 'box')

        for u,v in uv[:, :, catalog, 0, :]:
            # print("\n---")
            # print(u[:10])
            # print("*")
            # print(v[:10])
            # print("---")
            ax[catalog].scatter(u, v, c="b", s=1)
            ax[catalog].scatter(-u, -v, c="r", s=1)

    fig.savefig("ConstructVisibilities/results/uv_tracks.png")
    print("Done!")


def plot_bases(new_dataset, source_index):
    fig, ax = plt.subplots(2,3,figsize=(15,10))

    for i, cat in enumerate(["FIRST", "LOFAR", "NVSS"]):
        ax[0,i].imshow(new_dataset[source_index,0,i,:,:], aspect="auto", origin="lower")
        ax[0,i].set_title(cat+"-"+"Real")

        ax[1,i].imshow(new_dataset[source_index,1,i,:,:], aspect="auto", origin="lower")
        ax[1,i].set_title(cat+"-"+"Imaginary")

    fig.savefig("ConstructVisibilities/results/bases.png")


def construct_visibilities(X, names, save_plots=False):

    m_pixels, l_pixels = X.shape[1:3]
    deg_per_pixel = 0.000416666666666666 # as stated in headers of ALL fits files
    # deg_per_pixel = 100 * (1.0 / 3600.0)

    delta_u = 1.0 / ((deg_per_pixel * l_pixels) * (np.pi / 180.0)) 
    delta_v = 1.0 / ((deg_per_pixel * m_pixels) * (np.pi / 180.0))

    u_range = int(delta_u * l_pixels)
    v_range = int(delta_v * m_pixels)

    u_offset = u_range // 2
    v_offset = v_range // 2

    X_shifted = np.fft.ifftshift(X, axes=(1,2))

    X_inv_fourier = np.fft.ifft2(X_shifted, axes=(1,2)) * (delta_u * delta_v)

    X_fourier_shifted = np.fft.fftshift(X_inv_fourier, axes=(1,2))

    if save_plots:
        plot_fourier(X, X_shifted, X_inv_fourier, X_fourier_shifted, u_offset, v_offset)

    del X, X_shifted, X_inv_fourier

    enu = np.array([
        [25.095,    -9.095,     0.045],
        [90.284,    26.380,     -0.226],
        [3.985,     26.893,     0.0],
        [-21.605,   25.494,     0.019],
        [-38.272,   -2.592,     0.391],
        [-61.595,   -79.688,    0.702],
        [-87.988,   75.754,     0.138],
    ]) 
    # * 79 # Moving arrays to have a better telescope resolution

    array_latitude = lat_from_string(" 52d 54' 32.00'' ") # LOFAR Coords
    frequency_GHz = dict(FIRST=1.4, LOFAR=0.144, NVSS=1.4) # !FIRST is closer to 1.5GHz (~20cm) and NVSS claims 1.4GHz (~21cm)

    lambdas = [299_792_458 / (freq * (10**9)) for freq in frequency_GHz.values()]
    xyz, distances = get_XYZ(enu, array_latitude)

    baseline_args = np.argsort(distances)
    xyz = xyz[baseline_args]
    distances = distances[baseline_args]

    declinations = get_field_centre_declinations(names)

    hour_angle_range = np.array([np.linspace(np.radians(-3 * 15), np.radians(3 * 15), 6*60)]) # "one point is plotted every minute" - https://www.aanda.org/articles/aa/full_html/2013/08/aa20873-12/aa20873-12.html
    # hour_angle_range = np.array([np.linspace(np.radians(-3 * 15), np.radians(3 * 15), 100)])

    uv = get_uvw(xyz, lambdas, declinations, hour_angle_range)[:,:2] #removing w component (not needed for 2D)
    baselines, coords, catalogs, sources, timesteps = uv.shape

    new_dataset = np.zeros((sources, 2, catalogs, baselines * 2, timesteps))

    
    if save_plots:
        plot_uv_tracks(uv)
        source_plot = 0
        mask = np.zeros_like(X_fourier_shifted[source_plot,:,:,0], dtype=float)

    for source in range(sources):
        print(f"Source {source+1}/{sources}", end="\r")
        for cat in range(catalogs):
            for base in range(baselines):
                for t in range(timesteps):
                    
                    u, v = uv[base, :, cat, source, t]
                    
                    pixel_x = int((u_offset + u) / delta_u)
                    pixel_y = int((v_offset + v) / delta_v)
                    new_dataset[source, 0, cat, base*2, t] = X_fourier_shifted[source,:,:,cat].real[pixel_y, pixel_x]
                    new_dataset[source, 1, cat, base*2, t] = X_fourier_shifted[source,:,:,cat].imag[pixel_y, pixel_x]

                    if source == source_plot and save_plots:
                        mask[pixel_y, pixel_x] = None

                    pixel_x = int((u_offset + -u) / delta_u)
                    pixel_y = int((v_offset + -v) / delta_v)
                    new_dataset[source, 0, cat, base*2+1, t] = X_fourier_shifted[source,:,:,cat].real[pixel_y, pixel_x]
                    new_dataset[source, 1, cat, base*2+1, t] = X_fourier_shifted[source,:,:,cat].imag[pixel_y, pixel_x]

                    if source == source_plot and save_plots:
                        mask[pixel_y, pixel_x] = None
                    
    if save_plots:
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        extent = [-u_offset, u_offset, -v_offset, v_offset]
        ax[0].imshow(X_fourier_shifted[source_plot,:,:,0].real, aspect="auto", origin="lower", extent=extent)
        ax[0].imshow(mask, aspect="auto", origin="lower", cmap="hot", extent=extent)
        ax[1].imshow(X_fourier_shifted[source_plot,:,:,0].imag, aspect="auto", origin="lower", extent=extent)
        ax[1].imshow(mask, aspect="auto", origin="lower", cmap="hot", extent=extent)
        fig.savefig("ConstructVisibilities/results/overlayed_image.png")
    
    return new_dataset
    

def main():
    SAVE_DATASET = False
    SAVE_PLOTS = True
    directory = "ConstructData/results/"
    # clip = "_clip"
    # kind = "_train_val"

    for kind in ["_train_val", "_test", "_augmented"]:
        for clip in ["_clip", "_no_clip"]:

            if kind == "_augmented":
                names = np.load(directory + "names_train_val.npy")
                name_index = 0
                
                # Load Augmetated Data
                with open(directory + "augmented_images"+clip+".txt", "rb") as file:
                    X_aug = []
                    while True:
                        print(f"Loading Augmented Data...{name_index+1}/{len(names)}")
                        try:

                            X = np.load(file, allow_pickle=True)
                            # repeat names[name_index] X.shape[0] times
                            repeated_name = np.array([names[name_index]] * X.shape[0])
                            X_aug.append(construct_visibilities(X, repeated_name, save_plots=SAVE_PLOTS))
                            name_index += 1

                        except EOFError:
                            print("\nDone!")
                            break
                print("Saving Augmented Data...")
                if SAVE_DATASET:
                    with open("ConstructVisibilities/results/bases_augmented"+clip+".npy", "wb") as f:
                        for i in range(len(X_aug)):
                            np.save(f, X_aug[i])

            else:
                X = np.load(directory + "X" + kind + clip + ".npy")
                names = np.load(directory + "names" + kind + ".npy")
                new_dataset = construct_visibilities(X,names, save_plots=SAVE_PLOTS)
                plot_bases(new_dataset, source_index=0)
                if SAVE_DATASET:
                    np.save("ConstructVisibilities/results/bases" + kind + clip + ".npy", new_dataset)
                del new_dataset

if __name__ == "__main__": main()