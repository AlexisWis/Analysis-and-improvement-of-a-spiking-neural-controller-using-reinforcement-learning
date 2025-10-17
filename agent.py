import nest
import json
from typing import Tuple
import time
import numpy as np
from action_enum import AgentAction

def build_code():
    nest.set_verbosity("M_ERROR")

    from pynestml.codegeneration.nest_code_generator_utils import NESTCodeGeneratorUtils

    # generate and build code
    input_layer_module_name, input_layer_neuron_model_name = \
    NESTCodeGeneratorUtils.generate_code_for("ignore_and_fire_neuron.nestml")

    output_layer_module_name, output_layer_neuron_model_name, output_layer_synapse_model_name = \
        NESTCodeGeneratorUtils.generate_code_for("liu_pan_iaf_neuron.nestml",
                                                "liu_pan_neuromodulated_stdp_synapse.nestml",
                                                post_ports=["post_spikes"],
                                                logging_level="DEBUG",
                                                codegen_opts={"delay_variable": {"liu_pan_neuromodulated_stdp_synapse": "d"},
                                                            "weight_variable": {"liu_pan_neuromodulated_stdp_synapse": "w"}})
    
    return input_layer_module_name, \
            input_layer_neuron_model_name, \
            output_layer_module_name, \
            output_layer_neuron_model_name, \
            output_layer_synapse_model_name

class SpikingAgent():
    cycle_period = 10.   # [ms], corresponding to 2 physics steps
   
    def __init__(self, input_layer_module_name, \
            input_layer_neuron_model_name, \
            output_layer_module_name, \
            output_layer_neuron_model_name, \
            output_layer_synapse_model_name, \
            beta = 0.01, explore_period = 100, gamma = 0.98) -> None:

        # Optimization parameters:
        self.beta = beta # learning rate
        self.explore_period = explore_period # time where p_explore = 1
        self.gamma = gamma # discount factor

        # BOXES FROM LIU&PAN CODE
        self.x_thresholds = np.array([-2.4, -0.8, 0.8, 2.4])
        self.v_thresholds = np.array([float("-inf"), float("+inf")])
        
        self.theta_thresholds = np.array([-12, -5.738738738738739, -2.8758758758758756, 0., 2.8758758758758756, 5.738738738738739, 12])
        self.theta_thresholds = self.theta_thresholds / 180 * np.pi
        
        self.w_thresholds = np.array([float("-inf"), -103., -91.7, -80.2, -68.8, -57.3, -45.9, -34.3, -22.9, -11.5, 0.,
                                                      11.5, 22.9, 34.3, 45.9, 57.3, 68.8, 80.2, 91.7, 103., float("+inf")]) #open intervals ignored here
        self.w_thresholds = self.w_thresholds / 180 * np.pi
        
        #Boxes as suggested by Barto et al
        # self.x_thresholds = np.array([-2.4, -0.8, 0.8, 2.4])
        # self.theta_thresholds = np.array([-12, -6, -1, 0, 1, 6, 12])
        # self.theta_thresholds = self.theta_thresholds /180 * np.pi
        # self.v_thresholds = np.array([float("-inf"), -0.5, 0.5, float("+inf")]) #open intervals ignored here
        # self.w_thresholds = np.array([float("-inf"), -50, 50, float("+inf")]) #open intervals ignored here
        # self.w_thresholds = self.w_thresholds /180 * np.pi

        self.dimensions = (len(self.x_thresholds) - 1, len(self.theta_thresholds) - 1, len(self.v_thresholds) - 1, len(self.w_thresholds) - 1)

        print("Dimension of input space: " + str(self.dimensions))

        self.episode = 1

        # Network Parameters
        self.construct_neural_network(input_layer_module_name, \
            input_layer_neuron_model_name, \
            output_layer_module_name, \
            output_layer_neuron_model_name, \
            output_layer_synapse_model_name)
        self.Q_left = 0.
        self.Q_right = 0.
        self.Q_old = None #Set to None to skip first timestep without the need of knowing sim_timestep
        self.Q_value_scale = 0.1
        self.prev_spikes_left = 0
        self.prev_spikes_right = 0
        self.Wmax = 0.3
        self.Wmin = 0.005
        self.last_action_chosen = None
        self.p_explore = 1.0
        self.R = 1.  # reward -- always 1!

    def discretize(self, value, thresholds):
        for i, limit in enumerate(thresholds):
            if value < limit:
                return i - 1
        return -1

    def get_state_neuron(self, state) -> int:
        idx = 0
        thresholds = [self.x_thresholds, self.theta_thresholds, self.v_thresholds, self.w_thresholds]
        for dim, val, thresh in zip(self.dimensions, state, thresholds):
            i = self.discretize(val,thresh)
            if i == -1:
                return -1
            idx = idx * dim + i

        return idx
    
    def construct_neural_network(self, input_layer_module_name, \
            input_layer_neuron_model_name, \
            output_layer_module_name, \
            output_layer_neuron_model_name, \
            output_layer_synapse_model_name):
        nest.ResetKernel()
        nest.Install(input_layer_module_name)   # makes the generated NESTML model available
        nest.Install(output_layer_module_name)   # makes the generated NESTML model available
        nest.rng_seed = int(time.time())

        self.input_size = self.dimensions[0] * self.dimensions[1] * self.dimensions[2] * self.dimensions[3]
        self.input_population = nest.Create(input_layer_neuron_model_name, self.input_size)
    
        
        self.output_population_left = nest.Create(output_layer_neuron_model_name, 10)
        self.output_population_right = nest.Create(output_layer_neuron_model_name, 10)
        
        self.spike_recorder_input = nest.Create("spike_recorder")
        nest.Connect(self.input_population, self.spike_recorder_input)

        self.multimeter_left = nest.Create('multimeter', 1, {'record_from': ['V_m', 'g_e']})
        nest.Connect(self.multimeter_left, self.output_population_left)
        self.multimeter_right = nest.Create('multimeter', 1, {'record_from': ['V_m', 'g_e']})
        nest.Connect(self.multimeter_right, self.output_population_right)

        syn_opts = {"synapse_model": output_layer_synapse_model_name,
                    "weight": 0.1 + nest.random.uniform(min=0.0, max=1.0) * 0.02,
                    "beta": self.beta,
                    "tau_tr_pre": 20., # [ms]
                    "tau_tr_post": 20.,  # [ms]
                    "wtr_max": 0.1,
                    "wtr_min": 0.,
                    "dA_pre": 0.0001,
                    "dA_post": -1.05E-7}
        
        nest.Connect(self.input_population, self.output_population_left, syn_spec=syn_opts)
        nest.Connect(self.input_population, self.output_population_right, syn_spec=syn_opts)


        self.output_population_spike_recorder_left = nest.Create("spike_recorder")
        nest.Connect(self.output_population_left, self.output_population_spike_recorder_left)

        self.output_population_spike_recorder_right = nest.Create("spike_recorder")
        nest.Connect(self.output_population_right, self.output_population_spike_recorder_right)
        

    def choose_action(self, Q_left, Q_right) -> AgentAction:
        # Implemented according to scheme described in paper
        if self.episode < self.explore_period:
            self.p_explore = 1.0
        if np.random.random() < self.p_explore:
            p = 0.5
        else:
            p = np.exp(10.0 * (Q_right-Q_left)) #XXX THIS IS NOT A VALID DISTRIBUTION
        return AgentAction.RIGHT if np.random.random() < p else AgentAction.LEFT

    def compute_Q_values(self) -> None:
        r"""The output of the SNN is interpreted as the (scaled) Q values in the thesis, but their code
        interprets the difference of previous and current output spikes as the Q value."""
        spikes_left = self.output_population_spike_recorder_left.n_events
        spikes_right = self.output_population_spike_recorder_right.n_events
        self.Q_left = self.Q_value_scale * spikes_left
        self.Q_right = self.Q_value_scale * spikes_right

    # Update agent on failure/success
    def reset(self, is_success, is_evaluation = False) -> None:

        if not is_evaluation:
            if self.last_action_chosen == AgentAction.RIGHT:
                syn = nest.GetConnections(source=self.input_population, target=self.output_population_right)
            else:
                syn = nest.GetConnections(source=self.input_population, target=self.output_population_left)
            
            TD = 0 if is_success else -self.Q_old
            syn.w = np.clip(syn.w + np.array(syn.beta) * TD * np.array(self.prev_syn_wtr_right), self.Wmin, self.Wmax)

            self.p_explore *= 0.99 #0.99 from Liu&Pan code

        self.episode += 1


    def update(self, state: Tuple[float,float,float,float], is_evaluation: bool = False) -> Tuple[int, dict]:

        #Reset spike recorders
        self.output_population_spike_recorder_left.n_events = 0
        self.output_population_spike_recorder_right.n_events = 0

        # make the correct input neuron fire
        self.input_population.firing_rate = 0.
        neuron_id = self.get_state_neuron(state)
        
        # if state was a failure
        if neuron_id == -1:
            self.reset(is_success=False)
            return AgentAction.FAILURE
        
        self.input_population[neuron_id].firing_rate = 5000.
        
        # simulate for one cycle
        nest.Simulate(SpikingAgent.cycle_period)

        self.multimeter_left.n_events = 0

        self.compute_Q_values()

        # set new dopamine concentration on the synapses
        # PROBLEM: HOW DO WE HANDLE FAILURE? The physics simulation immediately resets after it.
        # Perhaps run the simulation without spiking to let the weights update? (BVogler)

        #Use Q_new because off-policy
        next_action = self.choose_action(self.Q_left, self.Q_right)
        Q_new = max(self.Q_left, self.Q_right)

        if self.Q_old is not None and not is_evaluation:
            TD = self.gamma * Q_new + self.R - self.Q_old
            
            if self.last_action_chosen == AgentAction.RIGHT:
                syn = nest.GetConnections(source=self.input_population, target=self.output_population_right)
                syn.w = np.clip(syn.w + np.array(syn.beta) * TD * np.array(self.prev_syn_wtr_right), self.Wmin, self.Wmax)
            else:
                assert self.last_action_chosen == AgentAction.LEFT
                syn = nest.GetConnections(source=self.input_population, target=self.output_population_left)
                syn.w = np.clip(syn.w + np.array(syn.beta) * TD * np.array(self.prev_syn_wtr_left), self.Wmin, self.Wmax)

        self.last_action_chosen = next_action

        if self.last_action_chosen == AgentAction.LEFT:
            self.Q_old = self.Q_left
        elif self.last_action_chosen == AgentAction.RIGHT:
            self.Q_old = self.Q_right

        #Note: Their code copies weight. traces after running the physics, but they only change during network simulation
        self.save_prev_syn_wtr()
        self.reset_prev_syn_wtr()

        return next_action
            
            
    def save_prev_syn_wtr(self):
        syn_right = nest.GetConnections(source=self.input_population, target=self.output_population_right)
        self.prev_syn_wtr_right = syn_right.wtr
        syn_left = nest.GetConnections(source=self.input_population, target=self.output_population_left)
        self.prev_syn_wtr_left = syn_left.wtr
    
    def reset_prev_syn_wtr(self):
        syn_to_left = nest.GetConnections(source=self.input_population, target=self.output_population_left)
        syn_to_right = nest.GetConnections(source=self.input_population, target=self.output_population_right)
        for _syn in [syn_to_left, syn_to_right]:
            _syn.wtr = 0.
            _syn.pre_trace = 0.
            #_syn.post_trace = 0. # need to do this in postsyn. neuron partner (lines below)

        self.output_population_left.post_trace__for_liu_pan_neuromodulated_stdp_synapse_nestml = 0.
        self.output_population_right.post_trace__for_liu_pan_neuromodulated_stdp_synapse_nestml = 0.