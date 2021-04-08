import hashlib
from lxml import etree
import subprocess as sub
import numpy as np
import os
from glob import glob

from cppn.tools.utilities import make_one_shape_only


def evaluate_population(pop, record_history=False):

    ret = True

    seed = pop.seed

    N = len(pop)
    if record_history:
        N = 1  # only evaluate the best ind in the pop
        # don't save the same design twice
        if os.path.isfile("a{0}_id{1}_fit{2}.hist".format(seed, pop[0].id, round(pop[0].fitness,2))):
            return

    # clear old .vxd robot files from the data directory
    sub.call("rm data{}/*.vxd".format(seed), shell=True)

    # remove old sim output.xml if we are saving new stats
    if not record_history:
        sub.call("rm output{}.xml".format(seed), shell=True)

    num_evaluated_this_gen = 0

    # hash all inds in the pop
    if not record_history:

        for n, ind in enumerate(pop):

            ind.teammate_ids = []
            ind.duplicate = False
            data_string = ""
            for name, details in ind.genotype.to_phenotype_mapping.items():
                data_string += details["state"].tostring()
                m = hashlib.md5()
                m.update(data_string)
                ind.md5 = m.hexdigest()

            if (ind.md5 in pop.already_evaluated) and len(ind.fit_hist) == 0:  # line 141 mutations.py clears fit_hist for new designs
                # print "dupe: ", ind.id
                ind.duplicate = True
            
            # It's still possible to get duplicates in generation 0.
            # Then there's two inds with the same md5, age, and fitness (because one will overwrite the other).
            # We can adjust mutations so this is impossible
            # or just don't evaluate th new yet duplicate design.
    
    # evaluate new designs
    for n, ind in enumerate(pop[:N]):

        # don't evaluate if invalid
        if not ind.phenotype.is_valid():
            for rank, goal in pop.objective_dict.items():
                if goal["name"] != "age":
                    setattr(ind, goal["name"], goal["worst_value"])

            print "Skipping invalid individual"

        # if it's a new valid design, or if we are recording history, create a vxd
        # new designs are evaluated with teammates from the entire population (new and old).
        elif (ind.md5 not in pop.already_evaluated) or record_history:

            num_evaluated_this_gen += 1
            pop.total_evaluations += 1

            (bx, by, bz) = ind.genotype.orig_size_xyz
            (wx, wy, wz) = ind.genotype.world_size

            root = etree.Element("VXD")  # new vxd root

            vxa_min_bot_size = etree.SubElement(root, "MinimumBotSize")
            vxa_min_bot_size.set('replace', 'VXA.Simulator.MinimumBotSize')
            vxa_min_bot_size.text = str(int(pop.min_bot_size))

            vxa_cilia_frac_after_first_round = etree.SubElement(root, "CiliaFracAfterFirstRound")
            vxa_cilia_frac_after_first_round.set('replace', 'VXA.Simulator.CiliaFracAfterFirstRound')
            vxa_cilia_frac_after_first_round.text = str(pop.cilia_frac_after_first_round)

            vxa_compute_largest_sticky_group_for_first_round = etree.SubElement(root, "ComputeLargestSitckyGroupForFirstRound")
            vxa_compute_largest_sticky_group_for_first_round.set('replace', 'VXA.Simulator.ComputeLargestSitckyGroupForFirstRound')
            vxa_compute_largest_sticky_group_for_first_round.text = str(int(pop.compute_largest_sticky_group_for_first_round))

            vxa_detach_stringy_bodies_every = etree.SubElement(root, "DetachStringyBodiesEvery")
            vxa_detach_stringy_bodies_every.set('replace', 'VXA.Simulator.DetachStringyBodiesEvery')
            vxa_detach_stringy_bodies_every.text = str(pop.detach_stringy_bodies_every)

            vxa_nonstick_time_after_stringy_body_detach = etree.SubElement(root, "nonStickyTimeAfterStringyBodyDetach")
            vxa_nonstick_time_after_stringy_body_detach.set('replace', 'VXA.Simulator.nonStickyTimeAfterStringyBodyDetach')
            vxa_nonstick_time_after_stringy_body_detach.text = str(pop.nonstick_time_after_stringy_body_detach)

            vxa_detach_probability = etree.SubElement(root, "DetachProbability")
            vxa_detach_probability.set('replace', 'VXA.Simulator.DetachProbability')
            vxa_detach_probability.text = str(pop.detach_probability)

            vxa_debris_spacing = etree.SubElement(root, "SpaceBetweenDebris")
            vxa_debris_spacing.set('replace', 'VXA.Simulator.SpaceBetweenDebris')
            vxa_debris_spacing.text = str(pop.space_between_debris)

            vxa_replenish_debris_every = etree.SubElement(root, "ReplenishDebrisEvery")
            vxa_replenish_debris_every.set('replace', 'VXA.Simulator.ReplenishDebrisEvery')
            vxa_replenish_debris_every.text = str(pop.replenish_debris_every)

            vxa_eval_period = etree.SubElement(root, "ReinitializeInitialPositionAfterThisManySeconds")
            vxa_eval_period.set('replace', 'VXA.Simulator.ReinitializeInitialPositionAfterThisManySeconds')
            vxa_eval_period.text = str(pop.eval_period)

            vxa_settle_time = etree.SubElement(root, "SettleTimeBeforeNextRoundOfReplication")
            vxa_settle_time.set('replace', 'VXA.Simulator.SettleTimeBeforeNextRoundOfReplication')
            vxa_settle_time.text = str(pop.settle_time)            

            vxa_debris_mat = etree.SubElement(root, "DebrisMat")
            vxa_debris_mat.set('replace', 'VXA.Simulator.DebrisMat')
            vxa_debris_mat.text = str(pop.debris_mat)

            vxa_debris_height = etree.SubElement(root, "DebrisHeight")
            vxa_debris_height.set('replace', 'VXA.Simulator.DebrisHeight')
            vxa_debris_height.text = str(pop.debris_height)

            vxa_debris_concentration = etree.SubElement(root, "DebrisConcentration")
            vxa_debris_concentration.set('replace', 'VXA.Simulator.DebrisConcentration')
            vxa_debris_concentration.text = str(int(pop.debris_concentration))

            vxa_world_size = etree.SubElement(root, "WorldSize")
            vxa_world_size.set('replace', 'VXA.Simulator.WorldSize')
            vxa_world_size.text = str(wx)

            vxa_wall_force = etree.SubElement(root, "WallForce")
            vxa_wall_force.set('replace', 'VXA.Simulator.WallForce')
            vxa_wall_force.text = str(pop.wall_force)

            vxa_randomize_cilia_every = etree.SubElement(root, "RandomizeCiliaEvery")
            vxa_randomize_cilia_every.set('replace', 'VXA.Simulator.RandomizeCiliaEvery')
            vxa_randomize_cilia_every.text = str(pop.randomize_cilia_every)

            # attach_detach = etree.SubElement(root, "AttachDetach")
            # attach_detach.set('replace', 'VXA.Simulator.AttachDetach')
            # etree.SubElement(attach_detach, "watchDistance").text = str(pop.attach_watch_dist)
            # etree.SubElement(attach_detach, "boundingRadius").text = str(pop.attach_bounding_radius)

            # set seed for browain cilia motion
            vxa_seed = etree.SubElement(root, "RandomSeed")
            vxa_seed.set('replace', 'VXA.Simulator.RandomSeed')
            vxa_seed.text = str(seed)

            if record_history:
                # sub.call("rm a{0}_gen{1}.hist".format(seed, pop.gen), shell=True)
                history = etree.SubElement(root, "RecordHistory")
                history.set('replace', 'VXA.Simulator.RecordHistory')
                etree.SubElement(history, "RecordStepSize").text = '100'
                etree.SubElement(history, "RecordVoxel").text = '1'
                etree.SubElement(history, "RecordLink").text = '1' if pop.wall_force > 0 else '1'
                etree.SubElement(history, "RecordFixedVoxels").text = str(int(pop.draw_walls))  # draw the walls of the dish
                etree.SubElement(history, "RecordCoMTraceOfEachVoxelGroupfOfThisMaterial").text = '0'  # record CoM

                # history = etree.SubElement(root, "Thermal")
                # history.set('replace', 'VXA.Environment.Thermal')
                # etree.SubElement(history, "TempPeriod").text = '0.1'

                # vxa_min_bot_size = etree.SubElement(root, "MinimumBotSize")
                # vxa_min_bot_size.set('replace', 'VXA.Simulator.MinimumBotSize')
                # vxa_min_bot_size.text = str(int(pop.min_bot_size * 2/3.0))  # lower the threshold for video


            structure = etree.SubElement(root, "Structure")
            structure.set('replace', 'VXA.VXC.Structure')
            structure.set('Compression', 'ASCII_READABLE')
            etree.SubElement(structure, "X_Voxels").text = str(wx)
            etree.SubElement(structure, "Y_Voxels").text = str(wy)
            etree.SubElement(structure, "Z_Voxels").text = str(wz)

            # If a single network output gives us the location and material of each voxel:
            if "Data" in ind.genotype.to_phenotype_mapping:
                for name, details in ind.genotype.to_phenotype_mapping.items():
                    state = details["state"]
                    flattened_state = state.reshape(wz, wx*wy)

                    data = etree.SubElement(structure, name)
                    for i in range(flattened_state.shape[0]):
                        if name == "Data":
                            layer = etree.SubElement(data, "Layer")
                            str_layer = "".join([str(c) for c in flattened_state[i]])
                            layer.text = etree.CDATA(str_layer)

            else: # otherwise we are doing swarm coevolution

                world = np.zeros((wx, wy, wz), dtype=np.int8)
                # world = np.zeros((wz, wx*wy), dtype=np.int)

                # now here is an interesting decision: should all debris share the same bias?
                # if not they will circle; could renormalize within a d_group; but for now:
                # bias = np.random.rand()
                # base_cilia_force = np.random.rand(wx, wy, wz, 3) - bias

                bodies = []
                
                for name, details in ind.genotype.to_phenotype_mapping.items():
                    if "body" in name:
                        this_bod = details["state"]

                        if np.sum(this_bod) > 0:
                            bodies += [details["state"]]

                if True:  # IDENTICAL_GEOMETRIES_IN_SWARM
                    bodies *= pop.swarm_size
                    # print len(bodies)

                # if False:

                #     if record_history:
                #         print "Recording a swarm with {} bodies".format(len(ind.best_teammates_so_far_ids))
                #         bodies += ind.best_teammates_so_far_data

                #     if len(bodies) < 2:
                #         rand_indices = np.random.permutation(len(pop))
                        
                #         count = -1
                #         while len(bodies) < 4:  # 4 teammates   

                #             count += 1                         

                #             this_teammate = pop[rand_indices[count]]
                            
                #             if this_teammate.duplicate:  # if it was just injected into pop but not novel
                #                 continue  # skip it

                #             bodies += [this_teammate.get_data("body")]   # add teammates to the swarm
                #             ind.teammate_ids += [this_teammate.id]  # this list was zeroed out above

                #             if count == len(pop)-1:
                #                 print "only {} full bodies in the whole pop".format(len(bodies))
                #                 break     

                
                # spacing between bodies:
                s = (wx - bx*pop.rows) / (pop.rows+1) #- (pop.rows%2==0)
                a = []
                b = []
                for r in range(pop.rows):
                    for c in range(pop.swarm_size/pop.rows):
                        a += [(r+1)*s+r*bx+int(wx%s>0)]
                        b += [(c+1)*s+c*bx+int(wx%s>0)]

                for n, (ai, bi) in enumerate(zip(a,b)):
                    try:
                        
                        world[ai:ai+bx, bi:bi+by, :bz] = bodies[n]

                        # base = np.random.rand(bx, by, bz, 3)
                        # base[:, :, :, 2] = np.zeros((bx, by, bz))

                        # # bias = np.random.rand()
                        # # these_forces = (base-np.min(base)) / (np.max(base)-np.min(base)) - bias
                        # these_forces = 2*base-1

                        # base_cilia_force[ai:ai+bx, bi:bi+by, :, :] = these_forces

                    except IndexError:
                        pass

                world = np.swapaxes(world, 0,2)
                # world = world.reshape([wz,-1])
                world = world.reshape(wz, wx*wy)

                if pop.wall_force == 0:
                    for i in range(wx):
                        for j in range(wy):
                            if (i == 0) or (j == 0) or (i == wx-1) or (j == wy-1):
                                world[:, i*wx+j] = 4  # wall

                def empty(i, j, k):
                    if pop.wall_force:
                        if (i <= 1) or (j <= 1) or (i >= wx-1) or (j >= wy-1):
                            return False

                    if ((world[k, i*wx+j] == 0) 
                    and (world[k-1, i*wx+j] in [0, pop.debris_mat]) and (world[k+1, i*wx+j] in [0, pop.debris_mat])
                    and (world[k, (i+1)*wx+j] == 0) and (world[k, (i-1)*wx+j] == 0) 
                    and (world[k, i*wx+j+1] == 0) and (world[k, i*wx+j-1] == 0) ):
                        return True
                    else:
                        return False


                bump = 0
                for i in range(0, wx, 5):
                    for j in range(0, wy, 5):
                        # if bump in [99]:
                        #     bump += 1
                        #     continue
                        if empty(i, j, 1) and empty(i, j-1, 1) and empty(i-1, j-1, 1) and empty(i-1, j, 1):
                            world[0, i*wx+j] = 3  # bump
                            bump += 1
            

                step = pop.space_between_debris + 1
                for i in range(2, wx, step): 
                    for j in range(2, wy, step):
                        k = pop.debris_height-1
                        try:
                            if empty(i, j, k):
                                world[k, i*wx+j] = pop.debris_mat  # pellet

                            if empty(i+step/2, j+step/2, k/2) and (pop.debris_concentration > 1):
                                world[k/2, (i+step/2)*wx+(j+step/2)] = pop.debris_mat  # pellet

                            if empty(i+step/2+1, j+step/2+1, 0) and (pop.debris_concentration > 2):
                                world[0, (i+step/2+1)*wx+(j+step/2+1)] = pop.debris_mat  # pellet

                            if empty(i+step/2+1, j+step/2+1, k+1) and (pop.debris_concentration > 3):
                                world[k+1, (i+step/2+1)*wx+(j+step/2+1)] = pop.debris_mat  # pellet

                        except IndexError:
                            pass

                data = etree.SubElement(structure, "Data")
                for i in range(world.shape[0]):
                    layer = etree.SubElement(data, "Layer")
                    str_layer = "".join([str(c) for c in world[i]])
                    layer.text = etree.CDATA(str_layer)

            # # evolved cilia forces
            # base_cilia_force = np.zeros((z, 3*x*y), dtype=np.float16)
            # for name, details in ind.genotype.to_phenotype_mapping.items():
            #     state = details["state"]
            #     flattened_state = state.reshape(z, x*y)

            #     if name == "cilia_X":
            #         base_cilia_force[:, ::3] = flattened_state
            #     if name == "cilia_Y":
            #         base_cilia_force[:, 1::3] = flattened_state
            #     # if name == "cilia_Z":
            #     #     base_cilia_force[:, 2::3] = flattened_state

            # data = etree.SubElement(structure, "BaseCiliaForce")
            # for i in range(base_cilia_force.shape[0]):
            #     layer = etree.SubElement(data, "Layer")
            #     str_layer = "".join([str(c) + ", " for c in base_cilia_force[i]])
            #     layer.text = etree.CDATA(str_layer)
            
            base_cilia_force = np.swapaxes(pop.base_cilia_force, 0,2)
            base_cilia_force = base_cilia_force.reshape(wz, 3*wx*wy)

            data = etree.SubElement(structure, "BaseCiliaForce")
            for i in range(base_cilia_force.shape[0]):
                layer = etree.SubElement(data, "Layer")
                str_layer = "".join([str(c) + ", " for c in base_cilia_force[i]])
                layer.text = etree.CDATA(str_layer)

            # save the vxd to data folder
            with open('data'+str(seed)+'/bot_{:04d}.vxd'.format(ind.id), 'wb') as vxd:
                vxd.write(etree.tostring(root))

    # ok let's finally evaluate all the robots in the data directory

    if record_history:  # just save history, don't assign fitness
        print "Recording the history of the run champ"
        sub.call("./voxcraft-sim -i data{0} > a{0}_id{1}_fit{2}.hist".format(seed, pop[0].id, round(pop[0].fitness,2)), shell=True)

    else:  # normally, we will just want to update fitness and not save the trajectory of every voxel

        print "GENERATION {}".format(pop.gen)

        print "Launching {0} voxelyze calls, out of {1} individuals".format(num_evaluated_this_gen, len(pop))

        if num_evaluated_this_gen > 0:

            while True:
                try:
                    sub.call("./voxcraft-sim -i data{0} -o output{1}.xml".format(seed, seed), shell=True)
                    # sub.call waits for the process to return
                    # after it does, we collect the results output by the simulator
                    root = etree.parse("output{}.xml".format(seed)).getroot()
                    break

                except IOError:
                    print "There was an IOError. Re-simulating this batch again..."
                    return False

                except IndexError:
                    print "There was an IndexError. Re-simulating this batch again..."
                    return False
            
            if len(glob("core.*")) > 0:
                sub.call("rm core.*", shell=True)

            # inds_to_update_ids = []

            for ind in pop:

                if ind.phenotype.is_valid() and ind.md5 not in pop.already_evaluated:

                    try:

                        sim_time = float(root.findall("detail/bot_{:04d}/currentTime".format(ind.id))[0].text)  # in seconds
                        largest_sticky_group = float(root.findall("detail/bot_{:04d}/largestStickyGroupSize".format(ind.id))[0].text)  # in voxels

                        rounds = int(sim_time / float(pop.eval_period))
                        # ind.pile_size = int(largest_sticky_group)
                        pile_size = largest_sticky_group / float(pop.min_bot_size)

                        ind.fitness = rounds + pile_size
                        
                        # num_links = float(root.findall("detail/bot_{:04d}/numRealLinks".format(ind.id))[0].text)  # at very end

                        # (wx, wy, wz) = ind.genotype.world_size
                        # n_vox = ((wx-2)/3.0)*((wy-2)/3.0)
                        # close_pairs_double_counted = float(root.findall("detail/bot_{:04d}/numClosePairs".format(ind.id))[0].text)
                        
                        # stuck_together_pairs = 0.5*close_pairs_double_counted/n_vox

                        # ind.fit_hist += [fit]  # history of every eval score with different teammates / cilia forces

                        # ind.best_score_so_far = fit
                        # ind.best_teammates_so_far_ids = ind.teammate_ids  # keep track of this for playback

                        # if False:

                        #     for tid in ind.teammate_ids:

                        #         this_teammate = pop.get_member_with_id(tid)
                        #         this_teammate.fit_hist += [fit]

                        #         ind.best_teammates_so_far_data += [this_teammate.get_data("body")]    # save for playback

                        #         if fit > this_teammate.best_score_so_far:  # new high score for teammate
                        #             this_teammate.best_score_so_far = fit
                        #             this_teammate.best_teammates_so_far_ids = [ind.id] + [tid2 for tid2 in ind.teammate_ids if tid2 != tid]

                        #             this_teammate.best_teammates_so_far_data =  [ind.get_data("body")]
                        #             for tid2 in ind.teammate_ids:
                        #                 if tid2 != tid:
                        #                     tm_tm = pop.get_member_with_id(tid2)
                        #                     this_teammate.best_teammates_so_far_data += [tm_tm.get_data("body")]

                        #         if tid not in inds_to_update_ids:
                        #             inds_to_update_ids += [tid]  # save so we can update each teammate's current fitness

                        # ind.fitness = np.mean(ind.fit_hist)  # current fitness


                        print "Assigning Ind {0}, Fit: {1}".format(ind.id, ind.fitness)

                        pop.already_evaluated[ind.md5] = [getattr(ind, details["name"])
                                                        for rank, details in
                                                        pop.objective_dict.items()]

                    except IndexError:
                        print "Couldn't find sim info for id {}. Re-simulating this batch".format(ind.md5)
                        ret = False
                        pass
        

            # for this_id in inds_to_update_ids:  # including already evaluated inds

            #     ind = pop.get_member_with_id(this_id)

            #     ind.fitness = np.mean(ind.fit_hist)

            #     print "id {0}: fit {1}, hist: {2}".format(this_id, round(ind.fitness, 2), ind.fit_hist)

            #     pop.already_evaluated[ind.md5] = [getattr(ind, details["name"])
            #                                       for rank, details in
            #                                       pop.objective_dict.items()]
                
    return ret


