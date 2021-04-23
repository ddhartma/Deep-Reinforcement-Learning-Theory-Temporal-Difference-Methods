[image1]: assets/grid_world_example.png "image1"
[image2]: assets/td_control_sarsa_1.png "image2"
[image3]: assets/sarsa_pseudocode.png "image3"
[image4]: assets/sarsamax.png "image4"
[image5]: assets/sarsamax_pseudocode.png "image5"
[image6]: assets/sarsamax_example.png "image6"
[image7]: assets/sarsa_example.png "image7"
[image8]: assets/expected_sarsa.png "image8"
[image9]: assets/pseudocode_expected_sarsa.png "image9"
[image10]: assets/expected_sarsa_example.png "image10"
[image11]: assets/cliff_walking.png "image11"
[image12]: assets/optimal_state_value_function.png "image12"
[image13]: assets/cliff_environment.png "image13"
[image14]: assets/sarsa_result.png "image14"
[image15]: assets/sarsamax_result.png "image15"
[image16]: assets/expected_sarsa_result.png "image15"

# Deep Reinforcement Learning Theory - Temporal Difference Methods

## Content
- [Introduction](#intro)
- [Temporal-Difference Methods](#tdm_overview)
- [TD Control](#TD_control)  
    - [Sarsa](#sarsa)
    - [Sarsamax  (or Q-Learning)](#sarsamax) 
    - [Expected Sarsa](#expected_sarsa)
- [TD Control: Theory and Practice](#TD_control_theory_practice) 
- [OpenAI Gym: CliffWalkingEnv](#CliffWalkingEnv)
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
- TD control can be used for continuous and episodic tasks
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

- Sarsa example:

    ![image7]

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

    ![image4]

- Sarsamax example:

    ![image6]

- Check out this [research paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.80.7501&rep=rep1&type=pdf) to read the proof that Q-Learning (or Sarsamax) converges.

### Pseudocode
- This is the Pseudocode for sarsamax

    ![image5]

## TD Control: Expected Sarsa <a name="expected_sarsa"></a>
- Expected Sarsa is an **on-policy TD control** method. It is guaranteed to converge to the optimal action value function **q∗**, under the same conditions that guarantee convergence of Sarsa and Sarsamax.

    ![image8]

- Expected Sarsa example:

    ![image10]

- Check out this [research paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.216.4144&rep=rep1&type=pdf) to learn more about Expected Sarsa.

### Pseudocode
- Pseudocode for Expected Sarsa

    ![image9]


## TD Control: Theory and Practice <a name="TD_control_theory_practice"></a> 
### Greedy in the Limit with Infinite Exploration (GLIE)
- The Greedy in the Limit with Infinite Exploration (GLIE) conditions were introduced in the previous lesson, when we learned about MC control. There are many ways to satisfy the GLIE conditions, all of which involve gradually decaying the value of ϵ\epsilonϵ when constructing ϵ\epsilonϵ-greedy policies.

- GLIE ensures that:
    - the **agent continues to explore** for all time steps, and
    - the **agent gradually exploits more** (and explores less).

- Both conditions are met if:
    - **ϵ<sub>i</sub> > 0** for all time steps **i**, and
    - **ϵ<sub>i</sub>** decays to zero in the limit as the time step **i** approaches infinity (**lim<sub> i→∞</sub> ϵ<sub>i</sub> = 0**).

### In Theory
- All of the **TD control algorithms** we have examined (Sarsa, Sarsamax, Expected Sarsa) are **guaranteed to converge to the optimal action-value function q∗**, as long as the step-size parameter **α** is sufficiently small, and the GLIE conditions are met.

- Once we have a good estimate for **q∗**, a corresponding optimal policy **π∗** can then be quickly obtained by setting **π∗(s) = argmax<sub>a∈A(s)</sub>q∗(s,a)** for all **s ∈ S**.

### In Practice
- In practice, it is common to completely ignore the GLIE conditions and still recover an optimal policy (see solution in notebook).

### Optimism
- Begin by initializing the values in the Q-table. It has been shown that initializing the estimates to large values can improve performance. For instance, if all of the possible rewards that can be received by the agent are negative, then initializing every estimate in the Q-table to zeros is a good technique. In this case, we refer to the initialized Q-table as **optimistic**, since the action-value estimates are guaranteed to be larger than the true action values.

## OpenAI Gym: CliffWalkingEnv <a name="CliffWalkingEnv"></a>
- The CliffWalking environment is a 4x12 gridworld matrix, with (using Numpy matrix indexing):
    - [3, 0] as the start at bottom-left
    - [3, 11] as the goal at bottom-right
    - [3, 1..10] as the cliff at bottom-center
- Each time step incurs -1 reward, and stepping into the cliff incurs -100 reward and a reset to the start. 
- An episode terminates when the agent reaches the goal.
- Please read about the cliff-walking task in Example 6.6 of the [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf). 
- Learn more about the environment in its corresponding [GitHub file](https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py)

    ![image11]

- The optimal state value function

    ![image12]

- Environment data

    ![image13]

### Implementation
- Open ```temporal_difference_methods.ipynb```
    Packages needed
    ```
    import sys
    import gym
    import numpy as np
    from collections import defaultdict, deque
    import matplotlib.pyplot as plt
    %matplotlib inline

    import check_test
    from plot_utils import plot_values
    ```
    Check the instance of the [CliffWalking](https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py) environment.
    ```
    env = gym.make('CliffWalking-v0')

    print(env.action_space)
    print(env.observation_space)

    RESULT:
    Discrete(4)
    Discrete(48)
    ```
    ### Update Q
    ```
    def update_Q(Qsa, Qsa_next, reward, alpha, gamma):
        """ Updates the action-value function estimate using the most recent time step 
            
            INPUTS:
            ------------
                Qsa - (float) action-value function for s_t, a_t
                Qsa_next - (float) action-value function for s_t+1, a_t+1
                reward - (int) reward for t+1
                alpha - (float) step-size parameter for the update step (constant alpha concept)
                gamma - (float) discount rate. It must be a value between 0 and 1, inclusive (default value: 1)
                
            OUTPUTS:
            ------------
                Qsa_update (float) updated action-value function for s_t, a_t
        """
        Qsa_update = Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))

        return Qsa_update
    ```
    ### Get epsilon greeed probabilities
    ```
    def epsilon_greedy_probs(env, Q_s, i_episode, eps=None):
        """ Obtains the action probabilities corresponding to epsilon-greedy policy 
            
            INPUTS:
            ------------
                env - (OpenAI gym instance) instance of an OpenAI Gym environment
                Q_s - (one-dimensional numpy array of floats) action value function for all four actions
                i_episode - (int) episode number
                eps - (float or None) if not None epsilon is constant
            
            OUTPUTS:
            ------------
                policy_s - (one-dimensional numpy array of floats) probability for all four actions, to get the most likely action
        
        """
        epsilon = 1.0 / i_episode
        if eps is not None:
            epsilon = eps
        policy_s = np.ones(env.nA) * epsilon / env.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / env.nA)
        return policy_s
    ```
    ### Sarsa
    ```
    def sarsa(env, num_episodes, alpha, gamma=1.0):
        """ TD Control: Sarsa
                
            INPUTS:
            ------------
                env - (OpenAI gym instance) instance of an OpenAI Gym environment
                num_episodes -(int) number of episodes that are generated through agent-environment interaction
                alpha - (float) step-size parameter for the update step (constant alpha concept)
                gamma - (float) discount rate. It must be a value between 0 and 1, inclusive (default value: 1)
                
            OUTPUTS:
            ------------
                Q - (dictionary of one-dimensional numpy arrays) where Q[s][a] is the UPDATED estimated action value 
                    corresponding to state s and action a
        """
        # initialize action-value function (empty dictionary of arrays)
        Q = defaultdict(lambda: np.zeros(env.nA))
        # initialize performance monitor
        plot_every = 100
        tmp_scores = deque(maxlen=plot_every)
        scores = deque(maxlen=num_episodes)
        # loop over episodes
        for i_episode in range(1, num_episodes+1):
            # monitor progress
            if i_episode % 100 == 0:
                print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
                sys.stdout.flush()   
            
            # initialize score
            score = 0
            # begin an episode, observe S
            state = env.reset()   
            # get epsilon-greedy action probabilities
            policy_s = epsilon_greedy_probs(env, Q[state], i_episode)
            # pick action A
            action = np.random.choice(np.arange(env.nA), p=policy_s)
            # limit number of time steps per episode
            for t_step in np.arange(300):
                # take action A, observe R, S'
                next_state, reward, done, info = env.step(action)
                # add reward to score
                score += reward
                if not done:
                    # get epsilon-greedy action probabilities
                    policy_s = epsilon_greedy_probs(env, Q[next_state], i_episode)
                    # pick next action A'
                    next_action = np.random.choice(np.arange(env.nA), p=policy_s)
                    # update TD estimate of Q
                    Q[state][action] = update_Q(Q[state][action], Q[next_state][next_action], 
                                                reward, alpha, gamma)
                    # S <- S'
                    state = next_state
                    # A <- A'
                    action = next_action
                if done:
                    # update TD estimate of Q
                    Q[state][action] = update_Q(Q[state][action], 0, reward, alpha, gamma)
                    # append score
                    tmp_scores.append(score)
                    break
            if (i_episode % plot_every == 0):
                scores.append(np.mean(tmp_scores))
                
        # plot performance
        plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False), np.asarray(scores))
        plt.xlabel('Episode Number')
        plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
        plt.show()
        # print best 100-episode performance
        print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))
        return Q
    ```
    ```
    # obtain the estimated optimal policy and corresponding action-value function
    Q_sarsa = sarsa(env, 5000, .01)

    # print the estimated optimal policy
    policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
    check_test.run_check('td_control_check', policy_sarsa)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_sarsa)

    # plot the estimated optimal state-value function
    V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
    plot_values(V_sarsa)
    ```
    ![image14]

    ### Sarsamax (Q-learning)
    ```
    def q_learning(env, num_episodes, alpha, gamma=1.0):
        # initialize empty dictionary of arrays
        """ TD Control: Sarsamax (Q-learning)
                
            INPUTS:
            ------------
                env - (OpenAI gym instance) instance of an OpenAI Gym environment
                num_episodes -(int) number of episodes that are generated through agent-environment interaction
                alpha - (float) step-size parameter for the update step (constant alpha concept)
                gamma - (float) discount rate. It must be a value between 0 and 1, inclusive (default value: 1)
                
            OUTPUTS:
            ------------
                Q - (dictionary of one-dimensional numpy arrays) where Q[s][a] is the UPDATED estimated action value 
                    corresponding to state s and action a
        """
        # initialize action-value function (empty dictionary of arrays)
        Q = defaultdict(lambda: np.zeros(env.nA))
        # initialize performance monitor
        plot_every = 100
        tmp_scores = deque(maxlen=plot_every)
        scores = deque(maxlen=num_episodes)
        # loop over episodes
        for i_episode in range(1, num_episodes+1):
            # monitor progress
            if i_episode % 100 == 0:
                print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
                sys.stdout.flush()   
            
            # initialize score
            score = 0
            # begin an episode, observe S
            state = env.reset()
            while True:
                # get epsilon-greedy action probabilities
                policy_s = epsilon_greedy_probs(env, Q[state], i_episode)
                # pick action A
                action = np.random.choice(np.arange(env.nA), p=policy_s)
                # take action A, observe R, S'
                next_state, reward, done, info = env.step(action)
                # add reward to score
                score += reward
                # update TD estimate of Q
                Q[state][action] = update_Q(Q[state][action], np.max(Q[next_state]), 
                                            reward, alpha, gamma)
                # S <- S'
                state = next_state
                # until S is terminal
                if done:
                    # append score
                    tmp_scores.append(score)
                    break
            if (i_episode % plot_every == 0):
                scores.append(np.mean(tmp_scores))
                
        # plot performance
        plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False), np.asarray(scores))
        plt.xlabel('Episode Number')
        plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
        plt.show()
        # print best 100-episode performance
        print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))
        return Q
    ```
    ```
    # obtain the estimated optimal policy and corresponding action-value function
    Q_sarsamax = q_learning(env, 5000, .01)

    # print the estimated optimal policy
    policy_sarsamax = np.array([np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4,12))
    check_test.run_check('td_control_check', policy_sarsamax)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_sarsamax)

    # plot the estimated optimal state-value function
    plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])
    ```
    ![image15]

    ### Expected Sarsa
    ```
    def expected_sarsa(env, num_episodes, alpha, gamma=1.0):
        # initialize empty dictionary of arrays
        """ TD Control: Expected Sarsa
                
            INPUTS:
            ------------
                env - (OpenAI gym instance) instance of an OpenAI Gym environment
                num_episodes -(int) number of episodes that are generated through agent-environment interaction
                alpha - (float) step-size parameter for the update step (constant alpha concept)
                gamma - (float) discount rate. It must be a value between 0 and 1, inclusive (default value: 1)
                
            OUTPUTS:
            ------------
                Q - (dictionary of one-dimensional numpy arrays) where Q[s][a] is the UPDATED estimated action value 
                    corresponding to state s and action a
        """
        # initialize action-value function (empty dictionary of arrays)
        Q = defaultdict(lambda: np.zeros(env.nA))
        # initialize performance monitor
        plot_every = 100
        tmp_scores = deque(maxlen=plot_every)
        scores = deque(maxlen=num_episodes)
        # loop over episodes
        for i_episode in range(1, num_episodes+1):
            # monitor progress
            if i_episode % 100 == 0:
                print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
                sys.stdout.flush()   
            
            # initialize score
            score = 0
            # begin an episode, observe S
            state = env.reset()
            # get epsilon-greedy action probabilities
            policy_s = epsilon_greedy_probs(env, Q[state], i_episode, 0.005)
            while True:
                # pick action A
                action = np.random.choice(np.arange(env.nA), p=policy_s)
                # take action A, observe R, S'
                next_state, reward, done, info = env.step(action)
                # add reward to score
                score += reward
                # get epsilon-greedy action probabilities (for S')
                policy_s = epsilon_greedy_probs(env, Q[next_state], i_episode, 0.005)
                # update TD estimate of Q
                Q[state][action] = update_Q(Q[state][action], np.dot(policy_s,Q[next_state]), 
                                            reward, alpha, gamma)
                # S <- S'
                state = next_state
                # until S is terminal
                if done:
                    # append score
                    tmp_scores.append(score)
                    break
            if (i_episode % plot_every == 0):
                scores.append(np.mean(tmp_scores))
                
        # plot performance
        plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False), np.asarray(scores))
        plt.xlabel('Episode Number')
        plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
        plt.show()
        # print best 100-episode performance
        print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))
        return Q
    ```
    ```
    # obtain the estimated optimal policy and corresponding action-value function
    Q_expsarsa = expected_sarsa(env, 10000, 1)

    # print the estimated optimal policy
    policy_expsarsa = np.array([np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4,12)
    check_test.run_check('td_control_check', policy_expsarsa)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_expsarsa)

    # plot the estimated optimal state-value function
    plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])
    ```
    ![image16]




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