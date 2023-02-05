# High-D Dataset Tools
As the name suggests, a set of tools designed to work with the High-D dataset (see https://www.highd-dataset.com/). Currently just a single script.

## Extract Two Agent Convoy Scenarios
Extracts two agent convoy scenarios from a set of High-D scenes. Uses a set of conditions in order to determine when two vehicles are in convoy and likely to exhibit causal behaviour before randomly selecting a vehicle in another lane to act as an independent agent. Outputs the causal scenarios as CSV formatted timeseries data.

    usage: extract_two_agent_convoy_scenarios.py [-h] [--all-kinematic-variables] [--interagent-distance-variables] [--ttc-variables] input_directory_path output_directory_path
    
Parameters:
* input_directory_path: Specifies path to directory containing High-D scenes to take as input.
* output_directory_path: Specifies path to directory to output two agent convoy scenarios to.
* -h: Displays the help message for the script.
* --all-kinematic-variables: Includes distance travelled and velocity for all scenario agents as variables in the output scenario. By default only includes acceleration for all scenario agent.
* --interagent-distance-variables: Includes distance between scenario agents as variables in the output scenario.
* --ttc-variables: Includes time to collision for all scenario agents as variables in the output scenario.