import sys, os
sys.path.append("../src")
from processing import data
from time import perf_counter as tick
import pandas as pd
import logging
import matplotlib.pyplot as plt

data.setup()
logging.basicConfig(level=logging.INFO, filename="src/tests/scalability.log", filemode="w", format="")
logging.info("Number of Sources,Time Taken")

# remove = []
# for k in range(10):

for num, n in enumerate(range(10, 100, 5)):

    num  += 1
    # num = k
    step = 360 / n
    types = ["FRI","FRII","COMPACT","BENT","X","RING","OTHER"]
    with open(f"src/tests/scalability{num}.csv", "w") as f:
        f.write("ra,dec,type")
        for i in range(0,n):
            f.write(f"\n{i*step},0,{types[i%7]}")


    start = tick()
    df = pd.read_csv(f"src/tests/scalability{num}.csv")
    data.newCatalog(df, f"scalability{num}", "ra", "dec", "type")
    data.analyse_new_catalog(f"scalability{num}")
    end = tick()
    data.rmCatalog([f"scalability{num}"])
    # remove.append(f"scalability{num}")
    logging.info(f"{n},{end-start}")
    # logging.info(f"{k},{end-start}")

    os.remove(f"src/tests/scalability{num}.csv")

# data.rmCatalog(remove)