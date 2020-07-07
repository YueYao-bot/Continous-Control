## Learning Algorithm
The learning algorithm here used is DDPG agent with the help of "Experience Replay" and "Fixed Targets". The actor network is a simple 3-layers fully connected network and only takes the state as input. The critic is also a simple 3-layers fully connected network. The architecture and activation function of critic are slightly different to the actor. And besides the state, critic will also take action into consideration.

Both actor and critic have a target network and those target networks will updated in *n* timesteps, where *n* is also a hyperparameter.

Hyperparameters are:
  
-  learning_rate: 0.001
- gamma: 0.99
- tau: 0.001
- Epsilon: 1.0
- Target networks of actor and critic update 10 times after each 20 timesteps.

## Plot of Scores
<img src="https://github.com/YueYao-bot/Udacity-Continous-Control/blob/master/Score_over_Episode.png"/>

Problem solved at 227 Episodes.

## Test Result
<img src="https://github.com/YueYao-bot/Udacity-Continous-Control/blob/master/test_result.gif"/>

Test Score: 26.6

## Ideas for Future Work
- More complex networks could be try out instead of fully connected network with only few layers.
- The performance of actor and critic with share weights will be interesting.
- The test result still don't reach to 30. More training episodes might improve the test result.