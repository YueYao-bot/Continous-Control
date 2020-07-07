# Project2: Continous Control


## Project Details
<img src="https://github.com/YueYao-bot/Udacity-Continous-Control/blob/master/test_result.gif"/>

For this project, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

Tested with python 3.6.0 together with

  * MacOS
  * pytorch 1.4.0
  * unityagents 0.4.0

## Getting Started
The required environment *Reacher20.app* is uploaded in this project. You can simply start the project without any additional work.



## Instructions
*Continous_Control.ipynb* is the entry of the project and has two modes. One for training and one for testing. Be sure that you have actor and critic networks of DDPG saved as *checkpoint_actor.pth* and *checkpoint_critic.pth*. Otherwise you need train the DDPG agent firstly. 

You can also use *Continous_Control.py* if you do not want to use jupyter notebook. It has same code as in *Continous_Control.ipynb*.

*checkpoint_actor.pth* and *checkpoint_critic.pth* save the weights of a trained model, you can load it and start the test mode!

This project solved the 2 second version of this problem, which means the *Reacher20.app* is the enviroment, where I trained and tested the DDPG agent