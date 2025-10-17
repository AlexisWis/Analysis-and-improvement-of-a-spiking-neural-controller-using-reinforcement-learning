from action_enum import AgentAction
from physics import Physics
from agent import SpikingAgent

def run_simulation(beta, explore_period, gamma, t, 
            input_layer_module_name, \
            input_layer_neuron_model_name, \
            output_layer_module_name, \
            output_layer_neuron_model_name, \
            output_layer_synapse_model_name):
    
    training_results = {}
    testing_results = {}
    params = str((beta, explore_period, gamma))
    training_results[params] = {}
    testing_results[params] = {}
    training_results[params], testing_results[params], visited_states = run_simulation_helper(beta, explore_period, gamma, 
                                    input_layer_module_name, \
                                    input_layer_neuron_model_name, \
                                    output_layer_module_name, \
                                    output_layer_neuron_model_name, \
                                    output_layer_synapse_model_name)
    return training_results, testing_results, visited_states
        


def run_simulation_helper(beta, explore_period, gamma,
                    input_layer_module_name, \
                    input_layer_neuron_model_name, \
                    output_layer_module_name, \
                    output_layer_neuron_model_name, \
                    output_layer_synapse_model_name):

    p = Physics()
    a = SpikingAgent(input_layer_module_name, \
                        input_layer_neuron_model_name, \
                        output_layer_module_name, \
                        output_layer_neuron_model_name, \
                        output_layer_synapse_model_name, \
                        beta, explore_period, gamma)
    
    max_episodes = 450
    max_timesteps = 200

    training_lifetime_log = []
    state_log = []

    for episode in range(max_episodes):

        for timestep in range(max_timesteps):

            state_log.append((a.discretize(p.get_state()[2], a.theta_thresholds), a.discretize(p.get_state()[3], a.w_thresholds)))
            # Run SNN simulation, returns action or failure
            action = a.update(p.get_state())
            if action == AgentAction.FAILURE:
                training_lifetime_log.append(timestep)
                break
            
            # Update physics with corresponding force direction
            force = 10 if action == AgentAction.RIGHT else -10
            p.update(force)

        else:   #Executes only if inner loop terminates successfully
            training_lifetime_log.append(199)
            a.reset(is_success=True)
        p.reset()
    
    #
    #   Testing phase
    #

    max_eval_episodes = 100

    testing_lifetime_log = []
    for episode in range(max_eval_episodes):

        for timestep in range(max_timesteps):

            # Run SNN simulation, returns action or failure
            action = a.update(p.get_state(), is_evaluation=True)
            if action == AgentAction.FAILURE:
                testing_lifetime_log.append(timestep)
                a.reset(is_success=False, is_evaluation=True)
                break
            
            # Update physics with corresponding force direction
            force = 10 if action == AgentAction.RIGHT else -10
            p.update(force)

        else:   #Executes only if inner loop terminates successfully
            testing_lifetime_log.append(199)
            a.reset(is_success=True, is_evaluation=True)
        p.reset()
    
    return training_lifetime_log, testing_lifetime_log, state_log
