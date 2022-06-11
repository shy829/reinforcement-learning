### Directory Structure

```
finalProject
|-- value_based
|	  |-- DDQN.py
|	  |-- Net.py
|	  |-- run.py
|	  |-- utils.py
|	  |__ result
|	      |-- DDQN_breakout.pkl
|	      |-- loss.txt
|	      |-- score_step.txt
|	      |__ score.txt
|
|-- policy_based
|   |-- Net.py
|   |-- ReplayBuffer.py
|   |-- run.py
|   |-- TD3.py
|   |-- utils.py
|   |-- model
|   |   |-- td3_actorHopper-v2
|   |   |-- td3_actorHopper-v2
|   |   |-- td3_actor_optimizerHopper-v2
|   |   |-- td3_actor_optimizerHumanoid-v2
|   |   |-- td3_criticHopper-v2
|   |   |-- td3_criticHumanoid-v2
|   |   |-- td3_critic_optimizerHopper-v2
|   |   |__ td3_critic_optimizerHumanoid-v2
|   |__ result
|       |-- Hopper-v2
|       |   |-- ep_reward.png
|       |   |-- ep_step.png
|       |   |__ ev_reward.png
|       |__ Humanoid-v2
|           |-- ep_reward.png
|           |-- ep_step.png
|           |__ ev_reward.png
|-- img
|-- report.md
|-- report.pdf
|-- lab_helper.md
|__ readme.md
```

#### How to run the project?

For example, if we need to run the value_based problem, open the terminal under the `value_based` package and then run the command `python run.py --env_name Humanoid-v2`. Policy_based problems can do the similar action. If you want if to run on the server and shows no output, you can see the lab_helper about how to run it.

#### The Package requirements

Python >= 3.7.0

Recommend to use conda to create a new environment for mujoco and atari. All the package can use the latest version. For atari, I use atari210 https://github.com/openai/atari-py#roms. Mujoco also can be installed by referencing https://github.com/openai/mujoco-py or other materials. I suggest that mujoco environment should be launched on Linux because windows only support version 1.5.0. 



