import numpy as np
from lxml import etree
import sys
import subprocess as sub
from glob import glob

PARENT_SWARM_ID = int(sys.argv[1])
THIS_SWARM_ID = int(sys.argv[2])
TREE_DEPTH = int(sys.argv[3])

# sub.call("rm voxcraft-sim && rm vx3_node_worker", shell=True)
# sub.call("cp /users/s/k/skriegma/sim/build/voxcraft-sim .", shell=True)
# sub.call("cp /users/s/k/skriegma/sim/build/vx3_node_worker .", shell=True)

DEBUG = False

SPAWN_CHILDREN_JOBS = False

partition = "dggpu"  # "dg-jup"

# debug
# if TREE_DEPTH > 4:
#     exit()

swarm_size = 9

SEED = THIS_SWARM_ID %(2**32-1) # Seed must be between 0 and 2**32 - 1
np.random.seed(SEED)

# MIN_BOT_SIZE = int(161*0.25)  # 40 (1/4 the original spherical workspace) 

sub.call("mkdir hist_data".format(THIS_SWARM_ID), shell=True)
sub.call("cp base.vxa hist_data/".format(THIS_SWARM_ID), shell=True)

sub.call("rm -r data{}".format(THIS_SWARM_ID), shell=True)
sub.call("mkdir data{}".format(THIS_SWARM_ID), shell=True)
sub.call("cp base.vxa data{}/base.vxa".format(THIS_SWARM_ID), shell=True)
sub.call("rm output{}.xml".format(THIS_SWARM_ID), shell=True)

EXTRA_WALL_HEIGHT = 16  # this is for structures above the body height; the wall force is infinitely high
WORLD_SIZE = (81, 81, 5+EXTRA_WALL_HEIGHT)
wx, wy, wz = WORLD_SIZE
BASE_CILIA_FORCE = np.zeros((wx, wy, wz, 3))  # np.ones((wx, wy, wz, 3)) * -1  # pointing downward
BASE_CILIA_FORCE[:, :, :, :2] = 2 * np.random.rand(wx, wy, wz, 2) - 1  # initial forces

if THIS_SWARM_ID == 0:
    CHILDREN = [161,]*9

else:
    root = etree.parse("output{}.xml".format(PARENT_SWARM_ID)).getroot()

    CHILDREN = []
    for pile in range(1, 19):
        try:
            child = int(root.findall("detail/swarm/pileSize{:02d}".format(pile))[0].text)
            if child > 0:
                CHILDREN += [child]
        except IndexError:
            pass

# CHILDREN = np.random.shuffle(CHILDREN)


def get_core_body_size(num_vox):

    if num_vox >= 100:  # 206:
        return (7, 7, 6)

    if num_vox >= 80:  # 161:
        return (7, 7, 5)

    if num_vox >= 60:  # 104:
        return (6, 6, 5)

    # if num_vox >= 70:  # 72:
    #     return (5, 5, 4)

    if num_vox >= 40: # 56:
        return (4, 4, 4)

    return (3, 3, 3)



def design(num_vox, orig_parent=False):

    if num_vox < 40:
        return np.zeros((7, 7, 6), dtype=np.int8)

    body_size = get_core_body_size(num_vox)
    l, w, h = body_size
    body = np.ones(body_size, dtype=np.int8)
    if w == 5:
        body = np.ones((l-1, w-1, h), dtype=np.int8)
    if w == 6:
        body = np.ones((l+1, w+1, h), dtype=np.int8)

    if w in [6, 7]:
        d = 7
        # spherification (in vivo bias; remove sharp corners):   
        sphere = np.zeros((d+2,)*3, dtype=np.int8) 
        radius = d/2+1
        r2 = np.arange(-radius, radius+1)**2
        dist2 = r2[:, None, None] + r2[:, None] + r2
        sphere[dist2 <= radius**2] = 1

        dz = 5
        for layer in range(5):
            if layer > dz/2:
                pad = 3
            else:
                pad = 1
            body[:, :, layer] = 1*sphere[1:d+1, 1:d+1, layer+pad]

        if w == 6:
            body[d/2:-1, :, :] = body[d/2+1:, :, :]
            body[-1, :, :] = 0
            body[:, d/2:-1, :] = body[:, d/2+1:, :]
            body[:, -1, :] = 0

        if h == 6:
            body[:, :, 3:] = body[:, :, 2:-1]
        else:
            body = np.pad(body, pad_width=((0, 0), (0, 0), (0, 6-h)), mode='constant', constant_values=0)

    elif w == 3:
        body = np.pad(body, pad_width=((2, 2), (2, 2), (0, 6-h)), mode='constant', constant_values=0)

    else:
        # smooth corners
        body[0, 0, 0] = 0
        body[-1, 0, 0] = 0
        body[0, -1, 0] = 0
        body[0, 0, -1] = 0
        body[0, -1, -1] = 0
        body[-1, -1, 0] = 0
        body[-1, 0, -1] = 0
        body[-1, -1, -1] = 0
        # make 7x7x6
        body = np.pad(body, pad_width=((1, 2), (1, 2), (0, 6-h)), mode='constant', constant_values=0)
        
        if w == 5:
            body[0, 2:4, 1:3] = 1
            body[-2, 2:4, 1:3] = 1
            body[2:4, 0, 1:3] = 1
            body[2:4, -2, 1:3] = 1
            # body[2:4, 2:4, -2] = 1

    max_size = np.sum(body>0)
        
    # print w, h, max_size
    # print max_size, np.sum(body)

    if orig_parent:
        body[l/2, w/2:, :] = 0  # evolved design

    return body 


root = etree.Element("VXD")  # new vxd root

# set seed for browain cilia motion
vxa_seed = etree.SubElement(root, "RandomSeed")
vxa_seed.set('replace', 'VXA.Simulator.RandomSeed')
vxa_seed.text = str(SEED)

# voxel data
structure = etree.SubElement(root, "Structure")
structure.set('replace', 'VXA.VXC.Structure')
structure.set('Compression', 'ASCII_READABLE')
etree.SubElement(structure, "X_Voxels").text = str(wx)
etree.SubElement(structure, "Y_Voxels").text = str(wy)
etree.SubElement(structure, "Z_Voxels").text = str(wz)

world = np.zeros((wx, wy, wz), dtype=np.int8)

c = len(CHILDREN)

if THIS_SWARM_ID == 0:
    bodies = [design(x, True) for x in CHILDREN]

else:
    if c == 0:
        exit()
    elif c == 1:
        bodies = [design(x) for x in CHILDREN]
    elif THIS_SWARM_ID % 2 != 0:
        bodies = [design(x) for x in CHILDREN[:c/2]]
    else:
        bodies = [design(x) for x in CHILDREN[c/2:]]

print "this swarm: ", [np.sum(x>0) for x in bodies]

rows = 3

bx, by, bz = 7, 7, 6

# spacing between bodies:
s = (wx - bx*rows) / (rows+1)
a = []
b = []
for r in range(rows):
    for c in range(rows):  # range(len(bodies)/rows)
        a += [(r+1)*s+r*bx+int(wx%s>0)]
        b += [(c+1)*s+c*bx+int(wx%s>0)]

for n, (ai, bi) in enumerate(zip(a,b)):
    try:
        world[ai:ai+bx, bi:bi+by, :bz] = bodies[n]
    except IndexError:
        pass

world[:2, :, :2] = 3  # wire
world[-2:, :, :2] = 3
world[:, :2, :2] = 3
world[:, -2:, :2] = 3

# world[:2, :2, 2] = 6  # holder
# world[:2, -2:, 2] = 6 
# world[-2:, :2, 2] = 6  
# world[-2:, -2:, 2] = 6

world[:2, :2, 2:6] = 5  # light
world[:2, -2:, 2:6] = 5 
world[-2:, :2, 2:6] = 5  
world[-2:, -2:, 2:6] = 5

for spot in [wx/2+1, wx/4+1, int(3*wx/4)+1]:
    world[:2, spot-4:spot+4, :2] = 0  # break line
    world[-2:, spot-4:spot+4, :2] = 0
    world[spot-4:spot+4, :2, :2] = 0 
    world[spot-4:spot+4, -2:, :2] = 0


world[:2, 35:37, 2:4] = 4  # battery
world[-2:, -36:-34, 2:4] = 4 

world = np.swapaxes(world, 0,2)
# world = world.reshape([wz,-1])
world = world.reshape(wz, wx*wy)

def empty(i, j, k):
    if (i <= 1) or (j <= 1) or (i >= wx-1) or (j >= wy-1):
        return False
    if ((world[k, i*wx+j] == 0) 
    and (world[k-1, i*wx+j] in [0, 2]) and (world[k+1, i*wx+j] in [0, 2])
    and (world[k, (i+1)*wx+j] == 0) and (world[k, (i-1)*wx+j] == 0) 
    and (world[k, i*wx+j+1] == 0) and (world[k, i*wx+j-1] == 0) ):
        return True
    else:
        return False


# n_bulbs = 0
component = 0
for i in range(16, wx-9, 17):
    for j in range(16, wy-9, 17):

        if component in [0, 3, 12, 15]:
            component += 1
            continue

        if empty(i, j, 1) and empty(i, j-1, 1) and empty(i-1, j-1, 1) and empty(i-1, j, 1) and \
           empty(i, j, 3) and empty(i, j-1, 3) and empty(i-1, j-1, 3) and empty(i-1, j, 3):

            rand_num = np.random.rand()

            # if component in [5, 6, 9, 10]:

            #     world[:2, i*wx+j-1:i*wx+j+1] = 3  # wire (sticky)
            #     world[:2, (i-1)*wx+j-1:(i-1)*wx+j+1] = 3

            #     if rand_num < 0.5:
            #         world[2:4, i*wx+j-1:i*wx+j+1] = 4  # battery
            #         world[2:4, (i-1)*wx+j-1:(i-1)*wx+j+1] = 4

            #     else:
            #         world[2:6, i*wx+j-1:i*wx+j+1] = 5  # light
            #         world[2:6, (i-1)*wx+j-1:(i-1)*wx+j+1] = 5
            #         # n_bulbs += 1

            if rand_num < 0.5:
                world[:2, i*wx+j-4:i*wx+j+4] = 3  # wire
                world[:2, (i-1)*wx+j-4:(i-1)*wx+j+4] = 3

            else:
                for adj in range(4):
                    world[:2, (i-adj)*wx+j-1:(i-adj)*wx+j+1] = 3
                    world[:2, (i+adj)*wx+j-1:(i+adj)*wx+j+1] = 3

        component += 1
            

step = 3  # pop.space_between_debris + 1
for i in range(2, wx, step): 
    for j in range(2, wy, step):
        k = 2  # debris_height-1
        try:
            if empty(i, j, k):
                world[k, i*wx+j] = 2  # pellet

            if empty(i+step/2, j+step/2, k/2):
                world[k/2, (i+step/2)*wx+(j+step/2)] = 2  # pellet

            if empty(i+step/2+1, j+step/2+1, 0):
                world[0, (i+step/2+1)*wx+(j+step/2+1)] = 2  # pellet

        except IndexError:
            pass

data = etree.SubElement(structure, "Data")
for i in range(world.shape[0]):
    layer = etree.SubElement(data, "Layer")
    str_layer = "".join([str(c) for c in world[i]])
    layer.text = etree.CDATA(str_layer)

vxa_base_cilia_force = np.swapaxes(BASE_CILIA_FORCE, 0,2)
vxa_base_cilia_force = vxa_base_cilia_force.reshape(wz, 3*wx*wy)

data = etree.SubElement(structure, "BaseCiliaForce")
for i in range(vxa_base_cilia_force.shape[0]):
    layer = etree.SubElement(data, "Layer")
    str_layer = "".join([str(c) + ", " for c in vxa_base_cilia_force[i]])
    layer.text = etree.CDATA(str_layer)

# save the vxd to data folder
with open('data{}/swarm.vxd'.format(THIS_SWARM_ID), 'wb') as vxd:
    vxd.write(etree.tostring(root))

# add tags for recording movie
history = etree.SubElement(root, "RecordHistory")
history.set('replace', 'VXA.Simulator.RecordHistory')
etree.SubElement(history, "RecordStepSize").text = '100'
etree.SubElement(history, "RecordVoxel").text = '1'
etree.SubElement(history, "RecordLink").text = '1'
etree.SubElement(history, "RecordFixedVoxels").text = '0'
etree.SubElement(history, "RecordCoMTraceOfEachVoxelGroupfOfThisMaterial").text = '0'  # record CoM

# save vxd to the hist_data folder
with open('hist_data/swarm{}.vxd'.format(THIS_SWARM_ID), 'wb') as vxd:
    vxd.write(etree.tostring(root))

if DEBUG:
    sub.call("rm a.hist", shell=True)
    sub.call("./voxcraft-sim -i hist_data > a.hist", shell=True)
    exit()

while True:
    try:
        sub.call("./voxcraft-sim -i data{0} -o output{0}.xml".format(THIS_SWARM_ID), shell=True)
        # sub.call waits for the process to return
        # after it does, we collect the results output by the simulator
        root = etree.parse("output{}.xml".format(THIS_SWARM_ID)).getroot()
        break

    except IOError:
        print "There was an IOError. Re-simulating this swarm again..."

    except IndexError:
        print "There was an IndexError. Re-simulating this swarm again..."

if len(glob("core.*")) > 0:
    sub.call("rm core.*", shell=True)


CHILDREN = []
for pile in range(1, 19):
    try:
        child = int(root.findall("detail/swarm/pileSize{:02d}".format(pile))[0].text)
        if child > 0:
            CHILDREN += [child]
    except IndexError:
        pass

try:
    lights_on = int(root.findall("detail/swarm/numLightsOn")[0].text)
    print "lights: ", lights_on
except IndexError:
    pass

print "children built: ", CHILDREN

tree_width_next_depth = 2**(TREE_DEPTH+1)
tree_width_this_depth = 2**TREE_DEPTH
child_id = (tree_width_next_depth - 1) + 2*THIS_SWARM_ID - 2*(tree_width_this_depth-1)

# print PARENT_SWARM_ID, THIS_SWARM_ID, child_id

if (THIS_SWARM_ID == 0):

    sub.call("sbatch submit_{0}.sh {1} {2} {3}".format(partition, THIS_SWARM_ID, 1, 1), shell=True)
    sub.call("sbatch submit_{0}.sh {1} {2} {3}".format(partition, THIS_SWARM_ID, 2, 1), shell=True)

elif SPAWN_CHILDREN_JOBS:

    if np.sum(CHILDREN) > 0:
        sub.call("sbatch submit_{0}.sh {1} {2} {3}".format(partition, THIS_SWARM_ID, child_id, TREE_DEPTH+1), shell=True)

    if np.sum(CHILDREN) > 1:
        sub.call("sbatch submit_{0}.sh {1} {2} {3}".format(partition, THIS_SWARM_ID, child_id+1, TREE_DEPTH+1), shell=True)

