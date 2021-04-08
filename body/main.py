import random
import numpy as np
import sys
from time import time
import cPickle
import subprocess as sub
from glob import glob
from functools import partial

from cppn.networks import CPPN, GeneralizedCPPN
from cppn.softbot import Genotype, Phenotype, Population
from cppn.tools.algorithms import Optimizer
from cppn.tools.utilities import natural_sort, make_one_shape_only, normalize, quadruped
from cppn.objectives import ObjectiveDict
from cppn.tools.evaluation import evaluate_population
from cppn.tools.mutation import create_new_children_through_mutation
from cppn.tools.selection import pareto_selection

# inputs:
# 1 = swarm size (ideally a perfect square: 4, 9, 16)
# 2 = eval time (fixed at 3; but ideally 2-5)
# 3 = seed (ideally 1-99)
# the actual SEED = 1000*size + 100*time + seed

DEBUG = False
reloadEx = False 
DRAW_WALLS = False  # todo: draw wall forces

SCULPT_Z_ONLY = True

SAVE_HIST_EVERY = 100  # gens

# if int(sys.argv[3]) == 99:  # quick test
#     SAVE_HIST_EVERY = 1  # save every new champ

GENS = 501
POPSIZE = 8*2-1 # +1 for the randomly generated robot that is added each gen

EVAL_PERIOD = 3 # int(sys.argv[2])  # need to adjust StopConditionFormula manually  3.5*4=14
SETTLE_TIME = 0.5
REPLENISH_DEBRIS_EVERY = EVAL_PERIOD + SETTLE_TIME

RANDMONIZE_CILIA_EVERY = 1.0

DETACH_STRINGY_BODIES_EVERY = 0.05
DETACH_PROBABILITY = 0.005  # gets multiplied by groupSize
NONSITCK_TIME_AFTER_STRINGY_BODY_DETACH = 0.06

COMPUTE_LARGEST_STICKY_GROUP_FOR_FIRST_ROUND_ONLY = False
CILIA_FRAC_AFTER_FIRST_ROUND = 4  # 0.008 vs 0.002 in first round
WALL_FORCE = 10

BODY_SIZE = (7, 7, 5)

SWARM_SIZE = int(sys.argv[1]) # this should be a perfect square: 4, 9, 16, 25, etc.
ROWS = int(np.sqrt(SWARM_SIZE))

DEBRIS_HEIGHT = 3
DEBRIS_CONCENTRATION = 3
SPACE_BETWEEN_DEBRIS = 2
DEBRIS_MAT = 2

MIN_BOT_SIZE = int(161*0.6667)  # 2/3 the workspace (sphere bounded)

WORLD_LEN = int(15*(SWARM_SIZE**0.5+1)+BODY_SIZE[0]*SWARM_SIZE**0.5) + int(SWARM_SIZE**0.5 % 2 == 0)
# print WORLD_LEN

SEED = SWARM_SIZE*1000 + EVAL_PERIOD*100 + int(sys.argv[3]) 
random.seed(SEED)
np.random.seed(SEED)

CHECKPOINT_EVERY = 1  # gens
MAX_TIME = 47  # [hours] evolution does not stop; after MAX_TIME checkpointing occurs at every generation.

if DEBUG:
    print "DEBUG MODE"
    sub.call("rm a{}_id0_fit-10000000.0.hist".format(SEED), shell=True)
    sub.call("rm -r pickledPops{0} && rm -r data{0}".format(SEED), shell=True)
    # WORLD_LEN = 105
    MIN_BOT_SIZE *= 0.5
    # EVAL_PERIOD *= 0.5
    # REPLENISH_DEBRIS_EVERY = EVAL_PERIOD + SETTLE_TIME

if reloadEx:
    print "UPDATING EXEC"
    sub.call("rm voxcraft-sim && rm vx3_node_worker", shell=True)
    sub.call("cp /users/s/k/skriegma/sim/build/voxcraft-sim .", shell=True)
    sub.call("cp /users/s/k/skriegma/sim/build/vx3_node_worker .", shell=True)

sub.call("mkdir pickledPops{}".format(SEED), shell=True)
sub.call("mkdir data{}".format(SEED), shell=True)
sub.call("cp base.vxa data{}/base.vxa".format(SEED), shell=True)

EXTRA_WALL_HEIGHT = 0  # unnecessary, the wall force is infinitely high
WORLD_SIZE = (WORLD_LEN, WORLD_LEN, BODY_SIZE[2]+EXTRA_WALL_HEIGHT)
wx, wy, wz = WORLD_SIZE
BASE_CILIA_FORCE = np.zeros((wx, wy, wz, 3))  # np.ones((wx, wy, wz, 3)) * -1  # pointing downward
BASE_CILIA_FORCE[:, :, :, :2] = 2 * np.random.rand(wx, wy, wz, 2) - 1  # initial forces

DIRECTORY = "."
start_time = time()


def body(data):
    l, w, h = BODY_SIZE
    design = np.greater(data, 0)  # output bw -/+1
    # body = make_one_shape_only(design)
    # body = body.astype(np.int8)
    body = design.astype(np.int8)

    if SCULPT_Z_ONLY:
        for layer in range(h):
            body[:, :, layer] = body[:, :, 0]

    # if DEBUG:
    #     body = np.ones(BODY_SIZE, dtype=np.int8)
    #     # body[l/2, w/2+1:, :] = 0

    # spherification (in vivo bias; remove sharp corners):   
    sphere = np.zeros((w+2,)*3, dtype=np.int8) 
    radius = w/2+1
    r2 = np.arange(-radius, radius+1)**2
    dist2 = r2[:, None, None] + r2[:, None] + r2
    sphere[dist2 <= radius**2] = 1

    max_size = 0
    for layer in range(h):
        if layer > h/2:
            pad = (h-1) - (w-h)/2
        else:
            pad = (w-h)/2
        body[:, :, layer] *= sphere[1:l+1, 1:w+1, layer+pad]
        max_size += np.sum(sphere[1:l+1, 1:w+1, layer+pad])

    # print max_size, np.sum(body)

    if np.sum(body) < int(max_size*0.3333):  # if there is too little material then discard the body
        # print "Disembodied!"
        return np.zeros(BODY_SIZE, dtype=np.int8)

    if (np.sum(body) > max_size - BODY_SIZE[2]):  # and (not DEBUG):  # if there is too little curvature then discard the body
        # print "Full; boring!"
        return np.zeros(BODY_SIZE, dtype=np.int8)

    else:
        while True:  # shift down until in contact with surface plane
            if np.sum(body[:, :, 0]) == 0:
                body[:, :, :-1] = body[:, :, 1:]
                body[:, :, -1] = np.zeros_like(body[:, :, -1])
            else:
                break
    
    body = make_one_shape_only(body)  # single contiguous body
    body = body.astype(np.int8)

    return body 


class MyGenotype(Genotype):

    def __init__(self):

        Genotype.__init__(self, orig_size_xyz=BODY_SIZE, world_size=WORLD_SIZE)

        self.add_network(CPPN(output_node_names=["body"]))
        self.to_phenotype_mapping.add_map(name="body", tag=None, func=body, output_type=int)

        # # what if we evolve a base and then slowly chip away at it with randomness?
        # self.add_network(CPPN(output_node_names=["cilia_X", "cilia_Y"]))
        # self.to_phenotype_mapping.add_map(name="cilia_X", tag="<cilia_X>", output_type=float)
        # self.to_phenotype_mapping.add_map(name="cilia_Y", tag="<cilia_Y>", output_type=float)


# Now specify the objectives for the optimization.
# Creating an objectives dictionary
my_objective_dict = ObjectiveDict()

# Adding an objective named "fitness", which we want to maximize.
# This information is returned by voxcraft-sim in a fitness .xml file, with a tag named "fitness_score"
my_objective_dict.add_objective(name="fitness", maximize=True, tag="<fitness_score>")

# # Add objective for rounds of replication
# my_objective_dict.add_objective(name="rounds", maximize=True, tag="<currentTime>")

# # Add objective to preserve largest piles, even if they do not make good F2 bots
# my_objective_dict.add_objective(name="pile_size", maximize=True, tag="<largestStickyGroupSize>")

# Add an objective to minimize the age of solutions: promotes diversity
my_objective_dict.add_objective(name="age", maximize=False, tag=None)


# Initializing a population of SoftBots
my_pop = Population(my_objective_dict, MyGenotype, Phenotype, pop_size=POPSIZE)
my_pop.seed = SEED
my_pop.swarm_size = SWARM_SIZE
my_pop.rows = ROWS
my_pop.min_bot_size = MIN_BOT_SIZE
my_pop.wall_force = WALL_FORCE
my_pop.compute_largest_sticky_group_for_first_round = COMPUTE_LARGEST_STICKY_GROUP_FOR_FIRST_ROUND_ONLY
my_pop.detach_stringy_bodies_every = DETACH_STRINGY_BODIES_EVERY
my_pop.nonstick_time_after_stringy_body_detach = NONSITCK_TIME_AFTER_STRINGY_BODY_DETACH
my_pop.detach_probability = DETACH_PROBABILITY
my_pop.eval_period = EVAL_PERIOD
my_pop.settle_time = SETTLE_TIME
my_pop.replenish_debris_every = REPLENISH_DEBRIS_EVERY
my_pop.debris_height = DEBRIS_HEIGHT
my_pop.debris_concentration = DEBRIS_CONCENTRATION
my_pop.space_between_debris = SPACE_BETWEEN_DEBRIS
my_pop.debris_mat = DEBRIS_MAT
my_pop.cilia_frac_after_first_round = CILIA_FRAC_AFTER_FIRST_ROUND
my_pop.randomize_cilia_every = RANDMONIZE_CILIA_EVERY
my_pop.base_cilia_force = BASE_CILIA_FORCE
my_pop.tiny_mode = False
my_pop.draw_walls = DRAW_WALLS

if DEBUG:
    # quick test to make sure evaluation is working properly:
    for ind in my_pop:
        evaluate_population(my_pop, record_history=True)
        my_pop.individuals = my_pop[1:]
    exit()

if len(glob("pickledPops{}/Gen_*.pickle".format(SEED))) == 0:
    # Setting up our optimization
    my_optimization = Optimizer(my_pop, pareto_selection, create_new_children_through_mutation, evaluate_population)

else:
    # continue from checkpoint
    successful_restart = False
    pickle_idx = 0
    while not successful_restart:
        try:
            pickled_pops = glob("pickledPops{}/*".format(SEED))
            last_gen = natural_sort(pickled_pops, reverse=True)[pickle_idx]
            with open(last_gen, 'rb') as handle:
                [optimizer, random_state, numpy_random_state] = cPickle.load(handle)
            successful_restart = True

            my_pop = optimizer.pop
            my_optimization = optimizer
            my_optimization.continued_from_checkpoint = True
            my_optimization.start_time = time()

            random.setstate(random_state)
            np.random.set_state(numpy_random_state)

            print "Starting from pickled checkpoint: generation {}".format(my_pop.gen)

        except EOFError:
            # something went wrong writing the checkpoint : use previous checkpoint and redo last generation
            sub.call("touch IO_ERROR_$(date +%F_%R)", shell=True)
            pickle_idx += 1
            pass


my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=GENS, 
                    checkpoint_every=CHECKPOINT_EVERY, save_hist_every=SAVE_HIST_EVERY, 
                    directory=DIRECTORY)


# print "That took a total of {} minutes".format((time()-start_time)/60.)
# # finally, record the history of best robot at end of evolution so we can play it back in VoxCad
# my_pop.individuals = [my_pop.individuals[0]]
# evaluate_population(my_pop, record_history=True)

