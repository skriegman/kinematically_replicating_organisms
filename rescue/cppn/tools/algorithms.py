import random
import time
import cPickle
import numpy as np
import subprocess as sub
import os


class Optimizer(object):
    def __init__(self, pop, selection_func, mutation_func, evaluation_func, num_rand_inds=1):
        self.pop = pop
        self.select = selection_func
        self.mutate = mutation_func
        self.evaluate = evaluation_func
        self.num_rand_inds = num_rand_inds
        self.continued_from_checkpoint = False
        self.start_time = None
        self.max_fitness = 1e10
        self.autosuspended = False

    def elapsed_time(self, units="s"):
        if self.start_time is None:
            self.start_time = time.time()
        s = time.time() - self.start_time
        if units == "s":
            return s
        elif units == "m":
            return s / 60.0
        elif units == "h":
            return s / 3600.0

    def save_checkpoint(self, directory, gen):
        sub.call("mkdir {0}/pickledPops{1}".format(directory, self.pop.seed), shell=True)

        random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        data = [self, random_state, numpy_random_state]

        with open('{0}/pickledPops{1}/Gen_{2}.pickle'.format(directory, self.pop.seed, gen), 'wb') as handle:
            cPickle.dump(data, handle, protocol=cPickle.HIGHEST_PROTOCOL)       

    def run(self, max_hours_runtime, max_gens, checkpoint_every, save_hist_every, directory="."):

        self.start_time = time.time()

        if self.autosuspended:
            sub.call("rm %s/AUTOSUSPENDED" % directory, shell=True)

        if not self.continued_from_checkpoint:  # generation zero

            while True:
                if self.evaluate(self.pop):
                    break

            self.select(self.pop)  # only produces dominated_by stats, no selection happening (population not replaced)

        while self.pop.gen < max_gens:

            if self.pop.gen % checkpoint_every == 0:  # and self.pop.gen > 0: 
                print "Saving checkpoint at generation {0}".format(self.pop.gen+1)
                self.save_checkpoint(directory, self.pop.gen)
                
            if self.pop.gen % save_hist_every == 0:
                if not os.path.isfile("a{0}_id{1}_fit{2}.hist".format(self.pop.seed, self.pop[0].id, int(100*self.pop[0].fitness))):
                    print "Saving history of run champ at generation {0}".format(self.pop.gen+1)
                    self.evaluate(self.pop, record_history=True)

            if self.elapsed_time(units="h") > max_hours_runtime or self.pop.best_fit_so_far == self.max_fitness:
                self.autosuspended = True
                print "Autosuspending at generation {0}".format(self.pop.gen+1)
                self.save_checkpoint(directory, self.pop.gen)
                # keep going but checkpoint every gen at this point
                # break

            self.pop.gen += 1

            # update ages
            self.pop.update_ages()

            # mutation
            print "Mutation starts"
            new_children = self.mutate(self.pop)
            print "Mutation ends: successfully generated %d new children." % (len(new_children))

            # combine children and parents for selection
            print "Now creating new population"
            self.pop.append(new_children)
            for _ in range(self.num_rand_inds):
                print "Random individual added to population"
                self.pop.add_random_individual()
            print "New population size is %d" % len(self.pop)

            # evaluate fitness
            print "Starting fitness evaluation"
            eval_timer = time.time()

            while True:
                if self.evaluate(self.pop):
                    break
                
            print "Fitness evaluation finished in {} seconds".format(time.time()-eval_timer)

            # perform selection by pareto fronts
            new_population = self.select(self.pop)

            # replace population with selection
            self.pop.individuals = new_population
            print "Population size reduced to %d" % len(self.pop)

            print "Non-dominated front:"
            for ind in self.pop:
                if len(ind.dominated_by) == 0:
                    print "ID {0}: age {1}, fitness {2}".format(ind.id, ind.age, ind.fitness)

        if not self.autosuspended:  # print end of run stats
            print "Finished {0} generations".format(self.pop.gen + 1)
            print "DONE!"

