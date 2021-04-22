[image1]: assets/grid_world_example.png "image1"
[image2]: assets/td_control_sarsa_1.png "image2"
[image3]: assets/sarsa_pseudocode.png "image3"

# Deep Reinforcement Learning Theory - Temporal Difference Methods

## Content
- [Introduction](#intro)
- [Temporal-Difference Methods](#tdm_overview)
- [TD Control](#TD_control)  
    - [Sarsa](#sarsa)
    - [Sarsamax  (or Q-Learning)](#sarsamax) 
    - [Expected Sarsa](#expected_sarsa)
- [Analyzing Performance](#analyze_perform)    
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)


## Introduction <a name="what_is_reinforcement"></a>
- Reinforcement learning is **learning** what to do — **how to map situations to actions** — so as **to maximize a numerical reward** signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. (Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html))
- Deep reinforcement learning refers to approaches where the knowledge is represented with a deep neural network

## Temporal-Difference Methods - Overview <a name="tdm_overview"></a>
- Real life is far from an episodic task 
- Whereas Monte Carlo (MC) prediction methods must wait until the end of an episode to calculate the return and to update the value function estimate, **temporal-difference (TD) methods update the value function after every time step**.
- For example **chess**: Agent will at every move be able to estimate the probability of winning the game. Monte Carlo instead needs the crah to learn anything.
- For example **self-driving cars**: Agent will be able to estimate if it's likely to crash
- TD control can be used for contnuous and episodic tasks
- This lesson covers material in Chapter 6 (especially 6.1-6.6) of the [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf).

## TD Control <a name="TD_control"></a>
- Remember the Constant-α MC control method:

    ![image1]

## TD Control: Sarsa <a name="sarsa"></a>
- Now let's update the Q-Table at the same time as the episode is unfolding.
The idea: 
- The current estimate for the value of selecting ***action right***
and ***state*** one is pulled from the Q-Table, it's just 6.
So, what about the alternative estimate?
- After we got the reward of negative one,
we ended up in ***state two*** and selected ***action right***.
- The Q-Table actually already has an estimate for the return: It's just the estimated action value for ***state two*** and ***action right***.
- So, our alternative estimate can just be ***7 = -1 + 8*** which is the value of the next state action pair.
- Then, just like in the Monte Carlo case, we can use this alternative estimate to update the Q-Table with a value between 6 and 7 depending on alpha. For alpha = 0.2 we get 6.2.
- We repeat the same process for the next time step where we update the entry in the Q-Table for ***state two*** and ***action right*** by just using the alternative estimate.
- The alternative estimate is just the reward we received
plus the currently estimated value of the next state action pair.
- For alpha = 0.2 we get 8.2.

    ![image2]

### Difference to MC?
- Instead of updating the Q-table **at the end of the epside** we update the Q-table **at every time step**.
- TD method can be used for episodic and continuous tasks.

### Pseudocode
- This is the Pseudocode for Sarsa

    ![image3]

- In the algorithm, the number of episodes the agent collects is equal to **num_episodes**. 
- For every time step **t ≥ 0**, the agent:

    - takes the action **A<sub>t</sub>** (from the current state **S<sub>t</sub>**) that is **ϵ**-greedy with respect to the Q-table,
    - receives the reward **R<sub>t+1</sub>** and next state **S<sub>t+1</sub>**,
    - chooses the next action **A<sub>t+1</sub>** (from the next state **S<sub>t+1</sub>**) that is **ϵ**-greedy with respect to the Q-table,
    - uses the information in the tuple (**S<sub>t</sub>**, **A<sub>t</sub>**, **R<sub>t+1</sub>**, **S<sub>t+1</sub>**, **A<sub>t+1</sub>**) to update the entry **Q(S<sub>t</sub>,A<sub>t</sub>)** in the Q-table corresponding to the current state **S<sub>t</sub>** and the action **A<sub>t</sub>**.

- **Sarsa(0)** (or **Sarsa**) is an **on-policy TD control** method. It is guaranteed to converge to the optimal action-value function **q∗**, as long as the step-size parameter **α** is sufficiently small and **ϵ** is chosen to satisfy the **Greedy in the Limit with Infinite Exploration (GLIE)** conditions.

## TD Control: Sarsamax  (or Q-Learning) <a name="sarsamax"></a>
- **Sarsamax** (or **Q-Learning**) is an **off-policy TD control** method. It is guaranteed to converge to the optimal action value function **q∗**, under the same conditions that guarantee convergence of the Sarsa control algorithm.

## TD Control: Expected Sarsa <a name="expected_sarsa"></a>
- Expected Sarsa is an **on-policy TD control** method. It is guaranteed to converge to the optimal action value function **q∗**, under the same conditions that guarantee convergence of Sarsa and Sarsamax.

## Analyzing Performance <a name="analyze_perform"></a>
- On-policy TD control methods (like Expected Sarsa and Sarsa) have better online performance than off-policy TD control methods (like Q-learning).
- Expected Sarsa generally achieves better performance than Sarsa.


## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Sparkify-Project.git
```

- Change Directory
```
$ cd Sparkify-Project
```

- Create a new Python environment, e.g. spark_env. Inside Git Bash (Terminal) write:
```
$ conda create --name spark_env
```

- Activate the installed environment via
```
$ conda activate spark_env
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
seaborn = 0.8.1
pyspark = 2.4.3
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Further Deep Reinforcement Learning References
* [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
* [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
* [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
* [Open AI Gym - Installation Guide](https://github.com/openai/gym#installation)
* [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)