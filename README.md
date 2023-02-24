# High-D Dataset Tools
As the name suggests, a set of tools designed to work with the High-D dataset (see https://www.highd-dataset.com/). Currently just a single script.

## Extract Two Agent Convoy Scenes
Extracts two agent convoy scenes from a set of High-D scenes. Uses a set of conditions in order to determine when two vehicles are in convoy and likely to exhibit causal behaviour before randomly selecting a vehicle in another lane to act as an independent agent.

```
extract_two_agent_convoy_scenes.py [-h] [--csv] [--json-meta] [--trimmed-scene-output-path TRIMMED_SCENE_OUTPUT_PATH] [--velocity-variables] [--all-kinematic-variables] [--interagent-distance-variables] [--ttc-variables] input_directory_path output_directory_path
```
    
Parameters:
* input_directory_path: Specifies path to directory containing High-D scenes to take as input.
* output_directory_path: Specifies path to directory to output two agent convoy scenes to.
* -h: Displays the help message for the script.
* --csv: Outputs the causal scenes as CSV formatted timeseries data.
* --json-meta: Outputs the causal scenes as JSON file with meta data.
* --trimmed-scene-output-path: Outputs the causal scenes as trimmed versions of the base High-D format to the specified output directory.
* --all-kinematic-variables: Includes distance travelled and velocity for all scenario agents as variables in the output scene. By default only includes acceleration for all scenario agents.
* --interagent-distance-variables: Includes distance between scenario agents as variables in the output scene.
* --ttc-variables: Includes time to collision for all scenario agents as variables in the output scene.
