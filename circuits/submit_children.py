import random
import numpy as np
import sys
from time import time
import cPickle
import subprocess as sub
from glob import glob
from lxml import etree


# only submits job for the last level
last_level = int(sys.argv[1])

total_nodes = 0
depth = 0
while depth <= last_level:
    total_nodes += 2**depth
    depth += 1

print "max tree size: ", total_nodes

highest_parent_node = (total_nodes-1) - 2**(depth-1)

print "highest parent: ", highest_parent_node

# print sub.check_output("squeue | grep skriegma | wc -l", shell=True)


def get_first_child_id_and_depth(this_id):
    nodes_so_far = 0
    depth = 0
    while True:
        if this_id > 2**depth + nodes_so_far - 1:
            nodes_so_far += 2**depth
            depth += 1
        else:
            # print this_id, depth
            break

    tree_width_next_depth = 2**(depth+1)
    tree_width_this_depth = 2**depth
    child_id = (tree_width_next_depth - 1) + 2*this_id - 2*(tree_width_this_depth-1)
    
    return child_id, depth


def already_evaluated(this_id):
    try:
        root = etree.parse("output{}.xml".format(this_id)).getroot()
        return True

    except IOError:
        return False


potential_parents = glob("output*.xml")
potential_parents_ids = [int(p[6:-4]) for p in potential_parents]

submitted = []
for swarm_id in potential_parents_ids:

    if swarm_id > highest_parent_node:
        continue

    # if swarm_id % 1e6 == 0:
    #     print "progress: ", round(1 - swarm_id/float(2**last_level-1), 4)

    n_children = 0
    first_child_id, this_depth = get_first_child_id_and_depth(swarm_id)

    # print swarm_id, first_child_id, this_depth+1
    # print swarm_id, first_child_id+1, this_depth+1

    try:
        root = etree.parse("output{}.xml".format(swarm_id)).getroot()

        for pile in range(1, 19):
            child = int(root.findall("detail/swarm/pileSize{:02d}".format(pile))[0].text)
            if child > 0:
                n_children += 1

        if n_children > 0 and not already_evaluated(first_child_id):
            submitted += [first_child_id]
            # print swarm_id, first_child_id, this_depth+1
            sub.call("sbatch submit_dggpu.sh {0} {1} {2}".format(swarm_id, first_child_id, this_depth+1), shell=True)

        if n_children > 1 and not already_evaluated(first_child_id+1):
            submitted += [first_child_id+1]
            # swarm_id, first_child_id+1, this_depth+1
            sub.call("sbatch submit_dggpu.sh {0} {1} {2}".format(swarm_id, first_child_id+1, this_depth+1), shell=True)

    except IOError:
        pass


print "no. of submissions: ", len(submitted)

exit()

