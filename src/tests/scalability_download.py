import sys, os
sys.path.append("../src")
from processing import data
from time import perf_counter as tick
import pandas as pd
import logging
import matplotlib.pyplot as plt

data.setup()
logging.basicConfig(level=logging.INFO, filename="src/tests/images.log", filemode="w", format="")
logging.info("Images,Time")

for k in range(1,11):
    print(f"--- {k} ---")
    for i in range(0,50,1):
        i += 1
        start = tick()
        data.download(
            cats = data.df["Catalog"].unique().tolist(),
            types = data.df["Type"].unique().tolist(),
            surveys =  ["NVSS"],
            image_limit=i,
            hasImage = True,
            threshold = None,
            prior = None,
        )
        end = tick()
        logging.info(f"{i},{end-start}")

    logging.info(f"---")