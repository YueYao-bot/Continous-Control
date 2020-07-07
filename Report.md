## Learning Algorithm
The learning algorithm here used is Deep Q Network with the help of "Experience Replay" and "Fixed Q Targets". The Q net is a simple 3-layers fully connected network.

Hyperparameters are:
  
-  learning_rate: 0.001
- gamma: 0.99
- tau: 0.001
- Epsilon: 1.0
- Target network of actor and critic update 10 times after each 20 timesteps.

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