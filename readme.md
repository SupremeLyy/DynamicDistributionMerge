# Improved Demonstration Knowledge Utilization in Reinforcement Learning

Hello, thanks for following our work!

This will help you understand and become familiar with our work Dynamic Distribution Merge.

All the requirement can be found by 

    require.ymal

And you only need to run this command to config the experiment environment.

    pip install ./require.ymal

If you are running the code for the first time, please create two new folders first

    mkdir runlog
    mkdir model

The experiment environment Maze Chase is in file mazeChase.py, you can see the demonstration by runing

    python mazeChase.py

These two folders store the run logs and the trained models. Then you can then run the following code directly to start training.

    python main.py


In main.py, we will first train a Q-Learning agent as a demonstrator. then we will use this agent as a demonstrator and test its performance in the environment. Finally, we train the agent using the DDM method. You can modify the parameter of training in the head of main.py

    alpha = 0.8
    gamma = 0.9
    epsilon = 0.8
    epsilonDecayStep = 3000     
    epsilonDecayStep episode
    epsilonDecayRatio = 0.95
    render = False
    trainEpisode = 260000+1 
    demonstratorEpisode = int(trainEpisode*0.5)
    maxStep = 1000
    sampleTrajectoryLength = 30000
    sampleTrajectoryErrorProb = 0.1
    OVERLAP_THRESHOLD = 0.45
    EXP_ENV = 'mazeChase'
    REWARD_NOISE = True       
    environmentChange = False

Especially, if want to evaluate that the Maze Chase Domain with Different Demonstration and
Learning Setting, you can set environmentChange to True

    environmentChange = True

In addition, we provide a test sample of time-varying MDP (TVMDP), where the transfer probability of the environment varies with the episodes of training sessions, which you can test by running the following code.

    python main-time-varying.py

Finally, you can see the training curve with tensorboard:

    tensorboard --logdir=runlog