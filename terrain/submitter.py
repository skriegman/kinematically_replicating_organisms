import time
import subprocess as sub

for cilia in range(50):
    print "sbatch submit.sh {}".format(cilia)
    sub.call("sbatch submit.sh {}".format(cilia), shell=True)
    time.sleep(0.1)

