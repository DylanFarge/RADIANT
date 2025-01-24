import psutil
import subprocess
import os
from time import sleep, time
from datetime import datetime

tick = time() / 3600
while True:

    count_procs = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3') - 1   # Child processes
    print("Number of processes: ", count_procs, " at ", datetime.now().strftime("%H:%M:%S"))

    if count_procs > 0:
        sleep(60 * 10) # 10 minutes
    
    else:
        count_items = 0

        for folder in os.listdir('/home/dylan/RADIANT/MachineLearning/results/'):
            count_items += len(os.listdir('/home/dylan/RADIANT/MachineLearning/results/' + folder))

        # if count_items == ( 34 * 35 ): #34 experiment folders * 35 files in each experiment folder
        if count_items == ( 48 * 35 ): #48 experiment folders * 35 files in each experiment folder
            print("All items are processed. Finished.")
            break 

        subprocess.call(['gnome-terminal', '--', 'python3','/home/dylan/RADIANT/MachineLearning/run_model.py'])
        print("New process started.")

print("Time taken: ", (time() / 3600) - tick, " hours")