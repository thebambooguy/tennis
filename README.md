[//]: # (Image References)

[image1]: https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png "Trained Agent"

# Tennis - Collaboration and Competition

## Introduction

Collaboration and Competition is the third and last project of a Deep Reinforcement Learning Nanodegree Program. 
For this project, an agent will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#tennis) environment. 
In this environment, two agents control rackets to bounce a ball over a net. Agents are rewarded every time they hit the 
ball over the net. Thus, the goal of each agent is to keep the ball in play.

![Trained Agent][image1]

## Project Details

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. 
Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement
toward (or away from) the net, and jumping. If an agent hits the ball over the net, it receives a reward of +0.1. 
If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 
(over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. 
    This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Getting Started

- **Step 1** - Set up your Python environment
  - Python 3.6 was used in the project
  - Type `pip install -r requirements.txt` in command line to install requirements
- **Step 2** - Build the Unity Environment 
  - Download the environment:
    - [Tennis](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
  - Unzip the environment and copy it to the main folder of the project. Note that provided environment will work only on Linux based machines!
- **Step 3** - Check `main.py` and familiarize yourself with arguments and hyper-parameters.
- **Step 4** - Run `main.py`
