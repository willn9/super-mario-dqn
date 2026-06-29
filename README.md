# Super Mario Bros Reinforcement Learning Agent

**Version:** 1.0  
**Author:** Wael Nawara  
**Framework:** Deep Q-Network (DQN) using PyTorch  
**Environment:** Super Mario Bros (NES) via Gym Super Mario Bros  
**Disclosure:** Gen AI (ChatGPT / Gemini etc.) were used in creating, debugging and improving many of the files as well as the documentation - so you may still find some bugs  or inconsistencies here and there - but I ran the code several times to remove as many bugs as I could.
---

# Project Overview

## Introduction

This project implements a complete **Deep Q-Network (DQN)** reinforcement learning agent capable of learning to play the original **Super Mario Bros** game. Rather than relying on an existing implementation, the project was developed incrementally from the ground up, with every major component designed, implemented, tested, and validated before being integrated into the final system.

The project serves two complementary purposes:

1. **A practical reinforcement learning implementation**, demonstrating how a Deep Q-Network can interact with a game environment and improve its performance through experience.

2. **A learning project**, documenting not only *what* was implemented, but also *why* each component exists and how the individual pieces work together.

The implementation follows the classical DQN approach introduced by DeepMind while emphasizing clean software engineering practices such as modularity, configurability, independent component testing, experiment tracking, and reproducibility.

---

## Project Objectives

The primary objectives of this project are:

* Build a complete reinforcement learning agent from scratch.
* Understand each component of the DQN algorithm rather than treating it as a black box.
* Create a modular project that can be extended with more advanced reinforcement learning algorithms in the future.
* Learn how reinforcement learning differs from supervised learning.
* Gain practical experience with PyTorch in a reinforcement learning setting.
* Develop a reproducible experimentation workflow including training logs, evaluation videos, plots, and configuration management.

---

## Why Super Mario Bros?

Super Mario Bros has become one of the classic benchmark environments for reinforcement learning because it combines many challenges encountered in real-world sequential decision problems.

Unlike simple control environments, Mario requires the agent to:

* interpret high-dimensional visual input,
* make decisions continuously,
* balance short-term and long-term rewards,
* navigate complex environments,
* avoid hazards,
* discover strategies through trial and error.

Although the environment appears simple to a human player, it presents a challenging learning problem for an artificial agent.

---

## Project Features

Version 1.0 includes:

### Environment

* Super Mario Bros NES environment
* Joypad action wrapper
* Frame skipping

### State Processing

* RGB to grayscale conversion
* Image resizing
* Pixel normalization
* Four-frame stacking

### Deep Reinforcement Learning

* Convolutional Neural Network (CNN)
* Deep Q-Network (DQN)
* Replay Memory
* ε-greedy exploration
* Bellman learning update
* Target Network synchronization

### Training

* GPU (CUDA) support
* Configurable hyperparameters
* Model checkpoint saving
* Training log generation
* Automatic learning curves

### Evaluation

* Separate evaluation script
* Model loading
* Gameplay video generation
* Frame capture
* Run metadata (`summary.json`)

---

## Software Engineering Goals

Beyond reinforcement learning itself, this project was intentionally designed to demonstrate good software engineering practices.

These include:

* Modular project organization
* Separation of training and evaluation
* Centralized configuration management
* Independent testing of individual components
* Experiment tracking
* Reproducibility
* Incremental development

The resulting project is considerably easier to understand, maintain, debug, and extend than a monolithic implementation.

---

## Learning Outcomes

Completing this project provides practical experience with:

* Reinforcement Learning fundamentals
* Deep Q-Networks
* Convolutional Neural Networks
* Experience Replay
* Target Networks
* PyTorch
* GPU acceleration using CUDA
* OpenAI Gym environments
* Experiment management
* Scientific visualization
* Modular software design

---

# Table of Contents

1. Project Overview

   * Introduction
   * Project Objectives
   * Why Super Mario Bros?
   * Project Features
   * Software Engineering Goals
   * Learning Outcomes

2. Environment Setup

   * Hardware
   * Software Requirements
   * Creating the Conda Environment
   * Installing Dependencies
   * CUDA Configuration
   * Version Compatibility
   * Common Installation Issues

3. Reinforcement Learning Overview

   * Agent
   * Environment
   * State
   * Action
   * Reward
   * Episode
   * Policy
   * Exploration vs. Exploitation

4. Overall Project Architecture

   * High-Level Pipeline
   * Data Flow
   * Training Workflow
   * Evaluation Workflow

5. Building the Project

   * Stage 1 – Environment Setup
   * Stage 2 – Frame Preprocessing
   * Stage 3 – Frame Stacking
   * Stage 4 – Deep Q-Network
   * Stage 5 – Replay Memory
   * Stage 6 – MarioAgent
   * Stage 7 – Training Loop
   * Stage 8 – Evaluation
   * Stage 9 – Logging
   * Stage 10 – Visualization

6. Project Directory Structure

7. File-by-File Description

   * `config.py`
   * `train.py`
   * `play.py`
   * `plot_training.py`
   * `mario_agent/agent.py`
   * `mario_agent/model.py`
   * `mario_agent/memory.py`
   * `mario_agent/wrappers.py`
   * `mario_agent/utils.py`

8. Main Classes and Functions

9. Configuration Management (`config.py`)

10. Testing Individual Components

    * `test_env.py`
    * `test_model.py`
    * `test_memory.py`
    * `test_agent.py`

11. Running the Complete Project

    * Training
    * Plot Generation
    * Evaluation

12. Output Files

    * Model Checkpoints
    * Training Log
    * Learning Curves
    * Gameplay Frames
    * Evaluation Video
    * Summary JSON

13. Lessons Learned

14. Future Enhancements

15. References


# 2. Environment Setup

A reproducible development environment is essential for reinforcement learning projects. This section documents the software, hardware, package versions, and installation steps used to develop this project.

Although the project is relatively small, several package compatibility issues were encountered during development, particularly involving **Gym**, **Gym Super Mario Bros**, **NumPy**, and **OpenCV**. The notes in this section are intended to help reproduce a working environment while avoiding those issues.

---

# 2.1 Hardware

The project was developed on the following hardware:

| Component        | Specification                  |
| ---------------- | ------------------------------ |
| Operating System | Windows 11 (64-bit)            |
| CPU              | AMD Ryzen 7 3700X              |
| RAM              | 32 GB                          |
| GPU              | NVIDIA GeForce RTX 2060 (6 GB) |
| CUDA Support     | Yes                            |

Although the project can run entirely on the CPU, GPU acceleration significantly reduces training time.

---

# 2.2 Software Requirements

The project uses the following software stack:

| Software             | Purpose                          |
| -------------------- | -------------------------------- |
| Python               | Programming language             |
| Conda                | Environment management           |
| PyTorch              | Deep Learning framework          |
| CUDA                 | GPU acceleration                 |
| Gym                  | Reinforcement learning interface |
| Gym Super Mario Bros | Game environment                 |
| NES-Py               | NES emulator backend             |
| OpenCV               | Image preprocessing              |
| NumPy                | Numerical computing              |
| Pandas               | Training log analysis            |
| Matplotlib           | Plot generation                  |

---

# 2.3 Creating the Conda Environment

Create a dedicated environment for the project.

```bash
conda create -n mario python=3.10
conda activate mario
```

Using a dedicated environment isolates project dependencies from other Python projects and simplifies troubleshooting.

---

# 2.4 Installing PyTorch

Install the CUDA-enabled version of PyTorch appropriate for your GPU.

Example:

```bash
pip install torch torchvision torchaudio
```

Verify the installation:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Expected output:

```
True
NVIDIA GeForce RTX 2060
```

---

# 2.5 Installing Project Dependencies

Install the remaining packages.

```bash
pip install gym
pip install gym-super-mario-bros
pip install nes-py
pip install opencv-python
pip install pandas
pip install matplotlib
```

Depending on the platform, some packages may already be installed as dependencies of others.

---

# 2.6 Verifying CUDA

Before beginning reinforcement learning experiments, verify that PyTorch is using the GPU.

```python
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)
```

Typical output:

```
cuda
```

If CUDA is unavailable, training will automatically fall back to the CPU.

---

# 2.7 Project Directory

The root directory contains the training scripts, evaluation scripts, configuration file, model checkpoints, logs, and source code.

Typical structure:

```
super_mario_learning/
```

Additional directories such as `logs/` and `runs/` are automatically created during execution.

---

# 2.8 Package Compatibility Notes

One of the most important lessons learned during development was that **package versions matter**.

Although newer versions of packages are generally desirable, older reinforcement learning environments sometimes depend on behaviors that have changed in recent releases.

The following combinations proved reliable during development:

| Package              | Recommended Version        |
| -------------------- | -------------------------- |
| Python               | 3.10                       |
| PyTorch              | CUDA-enabled build         |
| NumPy                | 1.26.4                     |
| OpenCV               | 4.11.x                     |
| Gym                  | Legacy Gym                 |
| Gym Super Mario Bros | Current compatible version |
| NES-Py               | Current compatible version |

Several issues were encountered while experimenting with different package versions, including:

* Gym deprecation warnings
* NumPy compatibility problems
* OpenCV and NumPy version interactions
* NES-Py integer overflow errors
* CUDA package compatibility

The project ultimately stabilized using **NumPy 1.26.4** together with **OpenCV 4.11**.

---

# 2.9 Common Warnings

During execution, several deprecation warnings may appear.

Examples include:

* Gym has been unmaintained since 2022.
* Old Step API warnings.
* Environment version recommendations.

These warnings do **not** prevent the project from running successfully and were expected during development.

Future versions of the project may migrate to **Gymnasium**, the successor to OpenAI Gym.

---

# 2.10 Verifying the Installation

Before beginning training, it is recommended to verify that the major components function correctly.

The project includes several standalone test programs:

```
python test_env.py
python test_model.py
python test_memory.py
python test_agent.py
```

Each test validates one subsystem independently before integrating the complete reinforcement learning pipeline.

This incremental testing strategy greatly simplified debugging during development and is recommended whenever modifications are made to the project.


# 3. Reinforcement Learning Overview

Before examining the implementation, it is useful to understand the reinforcement learning concepts that motivate the design of the project.

Unlike supervised learning, where a model learns from labeled examples, a reinforcement learning agent learns by **interacting with an environment**. The agent repeatedly observes its surroundings, takes actions, receives rewards, and gradually improves its decision-making strategy through experience.

The goal is not to imitate correct behavior, but to **discover** it through trial and error.

---

# 3.1 The Reinforcement Learning Loop

At every time step, the agent performs the following sequence:

1. Observe the current state.
2. Select an action.
3. Execute the action.
4. Receive a reward.
5. Observe the new state.
6. Store the experience.
7. Learn from previous experiences.
8. Repeat until the episode ends.

This interaction can be visualized as:

```text
                +------------------+
                |                  |
                |   Environment    |
                |                  |
                +---------+--------+
                          ^
                          |
                  Reward  |  Action
                          |
                          |
                +---------+--------+
                |                  |
                |      Agent       |
                |                  |
                +---------+--------+
                          |
                          |
                     Current State
```

The agent continually repeats this loop throughout training.

---

# 3.2 The Environment

The **environment** represents the external world in which the agent operates.

In this project, the environment is:

> **Super Mario Bros (NES)**

The environment is responsible for:

* Rendering the game.
* Updating Mario's position.
* Detecting collisions.
* Computing rewards.
* Ending an episode when Mario dies or completes the level.

The agent cannot directly modify the environment—it can only influence it by choosing actions.

---

# 3.3 The Agent

The **agent** is the decision-maker.

Its responsibilities include:

* observing the game state,
* selecting an action,
* storing experiences,
* learning from replay memory,
* updating the neural network,
* gradually improving its policy.

In this project, the agent is implemented by the `MarioAgent` class.

---

# 3.4 State

A **state** is the information available to the agent at a particular moment.

The raw game screen contains:

* Mario
* enemies
* platforms
* pipes
* coins
* background
* obstacles

Rather than using the full-color game image, the project preprocesses each frame by:

* converting to grayscale,
* resizing to 84 × 84 pixels,
* normalizing pixel values.

Because a single image cannot represent movement, the project stacks four consecutive frames.

The resulting state has dimensions:

```text
(4, 84, 84)
```

This allows the neural network to infer motion, speed, and direction.

---

# 3.5 Action

An **action** is one of the possible commands the agent can execute.

Examples include:

* move right,
* move left,
* jump,
* run,
* combinations of movement and jumping.

The available actions are determined by the **JoypadSpace** wrapper using the predefined `SIMPLE_MOVEMENT` action set.

At every time step, the neural network estimates the value of each possible action, and the agent selects one of them.

---

# 3.6 Reward

A **reward** is a numerical signal indicating how desirable an action was.

Typical rewards in Super Mario Bros include:

* moving forward,
* collecting coins,
* defeating enemies,
* completing a level.

Negative rewards may result from:

* losing time,
* dying,
* undesirable actions.

The objective of reinforcement learning is **not** to maximize the immediate reward, but rather the **total accumulated reward** over an entire episode.

---

# 3.7 Episode

An **episode** is one complete playthrough.

An episode begins when:

* the environment is reset.

It ends when:

* Mario dies,
* Mario completes the level,
* or the maximum number of allowed steps is reached.

Training consists of many consecutive episodes.

Each episode provides additional experience from which the agent learns.

---

# 3.8 Policy

A **policy** defines how the agent chooses actions.

Initially, the agent has no knowledge of the environment.

Consequently, early actions are largely exploratory.

As training progresses, the neural network gradually approximates a better policy by estimating the expected future reward associated with each action.

The policy therefore improves continuously throughout training.

---

# 3.9 Exploration vs. Exploitation

One of the central challenges in reinforcement learning is balancing:

* **Exploration** — trying new actions.
* **Exploitation** — selecting the action currently believed to be best.

If the agent always exploits its current knowledge, it may become trapped in a poor strategy.

If it always explores, it never takes advantage of what it has learned.

This project uses the classical **ε-greedy** strategy.

With probability **ε**, the agent selects a random action.

With probability **1 − ε**, the agent chooses the action with the highest predicted Q-value.

Initially:

```text
ε = 1.0
```

meaning nearly all actions are random.

As training progresses, ε gradually decreases until reaching a predefined minimum value.

This schedule allows the agent to explore extensively early in training while becoming increasingly deterministic as its policy improves.

---

# 3.10 Deep Q-Network (DQN)

The Deep Q-Network is the core learning algorithm used in this project.

Rather than storing a gigantic lookup table of state-action values, the DQN uses a **Convolutional Neural Network (CNN)** to approximate the **Q-function**.

The network receives the current state (four stacked frames) as input and predicts the expected long-term reward associated with every available action.

The agent then selects one of those actions according to the ε-greedy policy.

During training, the network continually updates its parameters so that its predictions become increasingly accurate.

---

# 3.11 Experience Replay

Learning directly from consecutive game frames is problematic because consecutive experiences are highly correlated.

To address this issue, the project stores experiences in a **Replay Memory**.

Each experience consists of:

* current state,
* chosen action,
* received reward,
* next state,
* episode completion flag.

During learning, the agent samples **random mini-batches** from memory.

Random sampling reduces correlation between training examples, improves stability, and increases data efficiency.

---

# 3.12 Target Network

One difficulty in reinforcement learning is that the neural network is simultaneously responsible for:

* predicting future rewards,
* and generating the targets it attempts to match.

This moving target can destabilize learning.

To address this problem, the project maintains two networks:

* **Online Network**
* **Target Network**

The online network is updated after every learning step.

The target network remains fixed and is synchronized only periodically.

This significantly improves training stability and is one of the defining characteristics of the original DQN algorithm.

---

# 3.13 Summary

The reinforcement learning concepts introduced in this chapter provide the foundation for the remainder of the project.

The following chapters explain how each of these ideas was translated into software, beginning with the overall architecture of the system and then progressing through the implementation of each component.

# 4. Overall Project Architecture

Having introduced the key reinforcement learning concepts, we can now examine how they are implemented in this project.

Rather than developing the system as a single large program, the project is organized into several independent components, each with a well-defined responsibility. This modular architecture simplifies development, testing, debugging, and future enhancements.

The overall processing pipeline is shown below.

---

# 4.1 High-Level Architecture

```text
                         ┌────────────────────┐
                         │ Super Mario Bros   │
                         │ Environment        │
                         └─────────┬──────────┘
                                   │
                          Raw RGB Frame
                                   │
                                   ▼
                     ┌────────────────────────┐
                     │ Frame Preprocessing    │
                     │ • Grayscale            │
                     │ • Resize (84×84)       │
                     │ • Normalize            │
                     └─────────┬──────────────┘
                               │
                               ▼
                     ┌────────────────────────┐
                     │ Frame Stack            │
                     │ (4 consecutive frames) │
                     └─────────┬──────────────┘
                               │
                               ▼
                     ┌────────────────────────┐
                     │ Deep Q-Network (CNN)   │
                     └─────────┬──────────────┘
                               │
                          Q-values
                               │
                               ▼
                     ┌────────────────────────┐
                     │ ε-Greedy Policy        │
                     └─────────┬──────────────┘
                               │
                            Action
                               │
                               ▼
                     ┌────────────────────────┐
                     │ Environment Step       │
                     └─────────┬──────────────┘
                               │
                 Reward • Next State • Done
                               │
                               ▼
                     ┌────────────────────────┐
                     │ Replay Memory          │
                     └─────────┬──────────────┘
                               │
                      Random Mini-batch
                               │
                               ▼
                     ┌────────────────────────┐
                     │ Bellman Learning       │
                     └─────────┬──────────────┘
                               │
                               ▼
                     Update Online Network
                               │
                               ▼
                  Periodically Update Target Network
```

This loop is repeated thousands of times during training.

---

# 4.2 Software Architecture

The project is divided into several logical modules.

```text
                train.py
                    │
                    ▼
             MarioAgent
                    │
      ┌─────────────┼──────────────┐
      │             │              │
      ▼             ▼              ▼
 ReplayMemory   MarioDQN      Wrappers
      │             │              │
      └─────────────┼──────────────┘
                    │
                    ▼
             Mario Environment
```

Each module has a single responsibility.

This separation makes the project easier to maintain and significantly reduces debugging effort.

---

# 4.3 Training Workflow

Training follows the sequence below.

```text
Initialize Environment
        │
        ▼
Initialize MarioAgent
        │
        ▼
Initialize Replay Memory
        │
        ▼
For each Episode
        │
        ▼
Reset Environment
        │
        ▼
Create Initial Frame Stack
        │
        ▼
Repeat
        │
        ▼
Select Action (ε-greedy)
        │
        ▼
Execute Action
        │
        ▼
Receive Reward
        │
        ▼
Update Frame Stack
        │
        ▼
Store Experience
        │
        ▼
Sample Replay Memory
        │
        ▼
Compute Bellman Target
        │
        ▼
Backpropagation
        │
        ▼
Update Online Network
        │
        ▼
Periodically Update Target Network
        │
        ▼
Until Episode Ends
```

At the end of every episode the project:

* records training statistics,
* appends a row to the CSV log,
* saves the latest model checkpoint,
* updates the exploration rate (ε).

---

# 4.4 Evaluation Workflow

Training and evaluation are intentionally separated.

The evaluation pipeline is considerably simpler.

```text
Load Trained Model
        │
        ▼
Reset Environment
        │
        ▼
Create Initial Frame Stack
        │
        ▼
Repeat
        │
        ▼
Select Best Action
 (ε = 0)
        │
        ▼
Execute Action
        │
        ▼
Save Selected Frames
        │
        ▼
Update Frame Stack
        │
        ▼
Until Episode Ends
        │
        ▼
Generate MP4 Video
        │
        ▼
Generate summary.json
```

Unlike training, evaluation performs **no learning**.

The model simply applies the policy learned during training.

---

# 4.5 Data Flow

One of the strengths of this project is that data moves through the system in a well-defined sequence.

```text
Raw RGB Image
        │
        ▼
Grayscale Image
        │
        ▼
84 × 84 Image
        │
        ▼
Normalized Image
        │
        ▼
Stack of Four Frames
        │
        ▼
CNN Input
        │
        ▼
Predicted Q-values
        │
        ▼
Selected Action
        │
        ▼
Environment Response
        │
        ▼
Replay Memory
        │
        ▼
Training Mini-batch
```

Understanding this flow greatly simplifies debugging because each processing stage can be verified independently.

---

# 4.6 Configuration Management

Rather than scattering constants throughout the source code, nearly all configurable parameters are centralized in a single file:

```text
config.py
```

Examples include:

* image size,
* frame stack size,
* frame skipping,
* learning rate,
* discount factor,
* batch size,
* epsilon schedule,
* target network update frequency,
* logging parameters,
* video generation parameters.

Centralizing these settings makes experimentation much easier and reduces the likelihood of inconsistent parameter values.

---

# 4.7 Experiment Outputs

Each complete experiment produces several artifacts.

```text
Training
    │
    ├── mario_dqn.pth
    ├── training.csv
    ├── reward plot
    ├── loss plot
    └── epsilon plot

Evaluation
    │
    ├── gameplay frames
    ├── run.mp4
    └── summary.json
```

These outputs provide both quantitative and qualitative evidence of the agent's performance.

---

# 4.8 Design Philosophy

Several software engineering principles guided the development of this project.

### Modular Design

Each file has a clearly defined responsibility.

---

### Incremental Development

Every component was implemented and tested independently before integration.

---

### Reproducibility

Configuration files, checkpoints, logs, plots, videos, and metadata allow experiments to be reproduced and compared.

---

### Separation of Concerns

Training, evaluation, visualization, preprocessing, and configuration are all handled by separate modules.

---

### Extensibility

The architecture was intentionally designed so that future improvements—such as Double DQN, Dueling DQN, or Prioritized Experience Replay—can be incorporated without major structural changes.

---

# 4.9 Summary

The project architecture reflects both reinforcement learning principles and good software engineering practices.

Instead of a monolithic implementation, the system is organized as a collection of focused, reusable components that communicate through well-defined interfaces. This modular approach made the project easier to develop incrementally, easier to test, and easier to understand.

The next chapter examines how the project itself was built, describing the implementation process stage by stage and explaining the purpose of each major component.

# 5. Building the Project

One of the primary goals of this project was to understand **how** a reinforcement learning system is constructed, rather than simply using an existing implementation.

For this reason, the project was developed incrementally. Each major component was implemented, tested independently, and validated before the next component was introduced. This approach significantly simplified debugging and ensured that each part of the system was understood before moving forward.

The following sections describe the development process in the order in which the project was built.

---

# Stage 1 — Creating the Environment

The first step was to establish communication with the Super Mario Bros game environment.

This involved:

* Creating the Gym environment.
* Wrapping the environment using `JoypadSpace`.
* Selecting the simplified action space (`SIMPLE_MOVEMENT`).
* Verifying that observations and rewards could be received.

At this stage, the agent had no intelligence. The objective was simply to ensure that the environment was functioning correctly.

### Main concepts introduced

* Environment creation
* Observation space
* Action space
* Reward signal
* Episode termination

### Validation

The environment was tested using:

```text
test_env.py
```

which confirmed:

* successful environment initialization,
* observation dimensions,
* available actions,
* reward values,
* information dictionary.

---

# Stage 2 — Frame Preprocessing

The raw game screen contains unnecessary information that increases computational cost.

Each frame was therefore transformed through several preprocessing steps:

1. Convert RGB image to grayscale.
2. Resize to **84 × 84** pixels.
3. Normalize pixel values to the range [0, 1].

These transformations reduce memory usage while preserving the visual information required for learning.

### Why grayscale?

Color is generally unnecessary for gameplay.

Removing color:

* reduces computation,
* reduces memory usage,
* simplifies learning.

### Why resize?

The original NES frame is considerably larger than necessary.

Reducing the resolution:

* accelerates training,
* decreases memory requirements,
* preserves important game features.

### Validation

The preprocessing pipeline was verified by examining:

* image dimensions,
* pixel ranges,
* data types.

---

# Stage 3 — Frame Stacking

A single image contains no information about movement.

For example:

* Is Mario jumping?
* Is an enemy moving left or right?
* Is Mario falling?

These questions cannot be answered from one frame alone.

To address this problem, the project stacks the four most recent processed frames.

The resulting state has dimensions:

```text
(4, 84, 84)
```

This allows the neural network to infer velocity and direction from consecutive observations.

### Validation

Frame stacking was tested by confirming:

* stack dimensions,
* frame ordering,
* update logic.

---

# Stage 4 — Building the Deep Q-Network

With the state representation complete, the next step was to construct the neural network.

The network consists of:

* convolutional layers,
* activation functions,
* fully connected layers,
* output layer producing one Q-value per action.

The input is the stacked game state.

The output is a vector of predicted Q-values.

For example:

```text
Action 0 → Q = 1.35
Action 1 → Q = 0.84
Action 2 → Q = 2.17
...
```

The action with the largest Q-value is considered the most promising.

### Validation

The network was verified by:

* creating dummy input tensors,
* checking tensor dimensions,
* confirming output shape.

---

# Stage 5 — Implementing Replay Memory

Learning from consecutive experiences is inefficient because neighboring game states are highly correlated.

Replay Memory addresses this issue by storing experiences for later reuse.

Each experience consists of:

* current state,
* chosen action,
* reward,
* next state,
* episode termination flag.

Experiences are stored in a fixed-size circular buffer.

During learning, random mini-batches are sampled.

### Advantages

Replay Memory:

* reduces correlation,
* improves stability,
* increases data efficiency,
* enables mini-batch learning.

### Validation

Replay Memory was tested by:

* storing experiences,
* checking memory size,
* sampling mini-batches,
* verifying returned data structures.

---

# Stage 6 — Building the MarioAgent

The `MarioAgent` class serves as the central controller of the project.

Rather than scattering functionality throughout the code, the agent encapsulates:

* neural network,
* target network,
* replay memory,
* optimizer,
* loss function,
* action selection,
* learning,
* checkpoint management.

The public interface of the agent is intentionally compact.

Typical methods include:

* `select_action()`
* `remember()`
* `learn()`
* `save()`
* `load()`
* `update_target_model()`

The remainder of the project interacts almost exclusively with these methods.

---

# Stage 7 — Implementing the Learning Algorithm

Once all supporting components were available, the learning algorithm could be implemented.

The learning process follows these steps:

1. Sample a random mini-batch.
2. Predict current Q-values.
3. Predict next-state Q-values.
4. Compute the Bellman targets.
5. Compute the loss.
6. Perform backpropagation.
7. Update network weights.

This process is repeated thousands of times during training.

The project uses:

* Huber Loss (`SmoothL1Loss`)
* Adam optimizer

Both are commonly used in Deep Q-Network implementations.

---

# Stage 8 — Training Loop

The training loop coordinates the entire learning process.

For each episode:

* reset the environment,
* create the initial frame stack,
* interact with the environment,
* store experiences,
* learn continuously,
* record statistics,
* save checkpoints.

Exploration gradually decreases using an ε-greedy schedule.

Periodic synchronization of the target network further stabilizes learning.

---

# Stage 9 — Evaluation Pipeline

Training and evaluation were deliberately separated.

The evaluation program:

* loads a trained model,
* disables exploration,
* plays the game,
* records selected frames,
* generates an MP4 video,
* saves metadata describing the run.

Keeping evaluation separate ensures that no learning occurs during gameplay visualization.

---

# Stage 10 — Experiment Tracking

A reinforcement learning project generates a considerable amount of information.

To make experiments reproducible, the project automatically records:

### Training

* model checkpoints,
* CSV logs,
* reward curves,
* loss curves,
* epsilon curves.

### Evaluation

* gameplay frames,
* MP4 videos,
* summary JSON files.

This information allows different experiments to be compared objectively and makes it easier to analyze improvements over time.

---

# Development Strategy

Looking back, the project followed a simple but effective strategy:

1. Build one component.
2. Test it independently.
3. Verify correctness.
4. Integrate it.
5. Repeat.

This incremental workflow greatly reduced debugging time.

Whenever a problem occurred, only the most recently added component needed to be investigated.

This approach proved far more manageable than attempting to build the entire reinforcement learning system in a single step.

---

# Summary

By the end of this stage-by-stage development process, the project had evolved from a simple game environment into a complete Deep Q-Network implementation capable of:

* learning from interaction,
* improving through experience,
* saving checkpoints,
* generating learning statistics,
* producing evaluation videos,
* documenting experiment results.

The next chapter examines the project structure in detail, describing the purpose of each directory, source file, class, and function.

# 6. Project Directory Structure

As the project evolved, the source code was organized into separate modules according to their responsibilities. This modular organization improves readability, simplifies testing, and makes future enhancements significantly easier.

The top-level project structure is shown below.

```text
super_mario_learning/
│
├── config.py
├── train.py
├── play.py
├── plot_training.py
│
├── mario_dqn.pth
│
├── logs/
│   ├── training.csv
│   ├── reward_vs_episode.png
│   ├── loss_vs_episode.png
│   └── epsilon_vs_episode.png
│
├── runs/
│   └── run_0001/
│       ├── frames/
│       ├── run.mp4
│       └── summary.json
│
├── mario_agent/
│   ├── __init__.py
│   ├── agent.py
│   ├── model.py
│   ├── memory.py
│   ├── wrappers.py
│   └── utils.py
│
├── test_env.py
├── test_model.py
├── test_memory.py
└── test_agent.py
```

The following sections describe the role of each component.

---

# 6.1 Root Directory

The root directory contains the project's main executable scripts together with the configuration file, model checkpoint, logs, and generated outputs.

| File               | Purpose                                 |
| ------------------ | --------------------------------------- |
| `config.py`        | Centralized project configuration       |
| `train.py`         | Trains the reinforcement learning agent |
| `play.py`          | Runs a trained agent without learning   |
| `plot_training.py` | Generates training plots                |
| `mario_dqn.pth`    | Latest trained model checkpoint         |

The root directory therefore acts as the entry point for the project.

---

# 6.2 The `mario_agent` Package

The `mario_agent` directory contains the core reinforcement learning implementation.

```text
mario_agent/
│
├── agent.py
├── model.py
├── memory.py
├── wrappers.py
├── utils.py
└── __init__.py
```

Each file has a single well-defined responsibility.

| File          | Responsibility                                            |
| ------------- | --------------------------------------------------------- |
| `agent.py`    | Reinforcement learning agent                              |
| `model.py`    | Neural network architecture                               |
| `memory.py`   | Replay memory                                             |
| `wrappers.py` | Image preprocessing and environment utilities             |
| `utils.py`    | General utility functions (reserved for future expansion) |
| `__init__.py` | Package initialization                                    |

---

# 6.3 The `logs` Directory

Training automatically creates the `logs` directory.

Example:

```text
logs/
│
├── training.csv
├── reward_vs_episode.png
├── loss_vs_episode.png
└── epsilon_vs_episode.png
```

## training.csv

Stores one row per episode.

Typical columns include:

* episode number
* reward
* loss
* epsilon
* replay memory size

This file represents the raw experimental data.

---

## Reward Plot

Visualizes how the episode reward changes throughout training.

Increasing reward generally indicates improved performance.

---

## Loss Plot

Shows how the neural network loss evolves.

Loss alone is not sufficient to evaluate reinforcement learning performance, but it is useful for detecting instability.

---

## Epsilon Plot

Shows the exploration schedule.

As training progresses, epsilon decreases from its initial value toward its minimum value.

---

# 6.4 The `runs` Directory

Each evaluation run automatically creates a new directory.

Example:

```text
runs/
│
└── run_0001/
```

This prevents previous evaluation results from being overwritten.

Each run contains:

```text
run_0001/
│
├── frames/
├── run.mp4
└── summary.json
```

---

## frames/

Contains selected gameplay frames captured during evaluation.

The frame frequency is configurable.

Each frame includes an overlay containing useful information such as:

* current step
* accumulated reward
* action
* Mario's X position

---

## run.mp4

Generated automatically after evaluation.

The video provides a qualitative assessment of the learned policy.

---

## summary.json

Stores metadata describing the evaluation run.

Typical information includes:

* checkpoint used
* number of steps
* final reward
* final X position
* timestamp
* configuration values

This makes individual evaluation runs reproducible.

---

# 6.5 Test Programs

The project includes several standalone test programs.

```text
test_env.py
test_model.py
test_memory.py
test_agent.py
```

Each validates one subsystem independently.

This modular testing strategy proved invaluable throughout development.

---

# 6.6 Configuration File

The project contains a single configuration file:

```text
config.py
```

Nearly all configurable parameters are stored here.

Examples include:

* frame preprocessing
* training hyperparameters
* exploration schedule
* target network updates
* logging
* evaluation

Centralizing configuration greatly simplifies experimentation.

---

# 6.7 Model Checkpoint

The trained neural network is stored in:

```text
mario_dqn.pth
```

This file contains the learned network parameters.

It can be:

* loaded for evaluation,
* used to resume training,
* archived for comparison with future models.

---

# 6.8 Separation of Responsibilities

One of the design goals of the project was that each file should perform one primary task.

The resulting organization is summarized below.

| Component          | Responsibility               |
| ------------------ | ---------------------------- |
| `config.py`        | Configuration                |
| `train.py`         | Learning                     |
| `play.py`          | Evaluation                   |
| `plot_training.py` | Visualization                |
| `agent.py`         | Reinforcement learning logic |
| `model.py`         | Neural network               |
| `memory.py`        | Experience replay            |
| `wrappers.py`      | Environment preprocessing    |
| `logs/`            | Training outputs             |
| `runs/`            | Evaluation outputs           |

This separation reduces coupling between modules and makes the project easier to understand.

---

# 6.9 Design Rationale

A common mistake in reinforcement learning projects is to place all logic inside a single training script.

Instead, this project separates:

* environment handling,
* preprocessing,
* learning,
* evaluation,
* configuration,
* visualization,
* testing.

This modular architecture provides several advantages:

* easier debugging,
* simpler maintenance,
* improved readability,
* better extensibility,
* easier experimentation.

Future enhancements can therefore be implemented with minimal impact on existing code.

---

# Summary

The project directory reflects the overall software architecture introduced in the previous chapter.

Rather than being organized around algorithms alone, the codebase is structured around responsibilities. Training, evaluation, visualization, configuration, preprocessing, and learning are all separated into dedicated modules, making the project easier to develop, test, and extend.

The next chapter examines each source file in detail, describing its purpose, major classes, key functions, and interactions with the remainder of the system.


# 7. File-by-File Description

This chapter describes the purpose of every source file in the project. Rather than focusing on implementation details, the objective is to explain the role of each file, its primary classes and functions, and how it interacts with the remainder of the system.

---

# 7.1 `config.py`

## Purpose

The `config.py` file centralizes nearly all configurable parameters used throughout the project.

Instead of scattering constants across multiple source files, all tunable values are defined in one location. This greatly simplifies experimentation and helps maintain consistency.

---

## Typical Parameters

### Environment

* `FRAME_SKIP`
* `STACK_SIZE`
* `IMAGE_SIZE`

---

### Learning

* `GAMMA`
* `LEARNING_RATE`
* `BATCH_SIZE`

---

### Training

* `NUM_EPISODES`
* `MAX_STEPS`

---

### Exploration

* `EPSILON_START`
* `EPSILON_MIN`
* `EPSILON_DECAY`

---

### Target Network

* `TARGET_UPDATE_EVERY`

---

### Logging

* `LOGS_DIR`
* `TRAINING_LOG_FILE`

---

### Evaluation

* `SAVE_EVERY_N_STEPS`
* `VIDEO_FPS`
* `CHECKPOINT_FILE`

---

## Why centralize configuration?

Centralized configuration offers several advantages:

* easier experimentation,
* improved consistency,
* cleaner code,
* simpler debugging,
* reproducible experiments.

Changing a parameter in one location automatically updates the behavior of the entire project.

---

# 7.2 `train.py`

## Purpose

`train.py` is the primary training program.

It coordinates the complete reinforcement learning process, including:

* environment creation,
* agent initialization,
* interaction with the environment,
* learning,
* logging,
* checkpoint generation.

---

## Main Workflow

1. Create the environment.
2. Create the agent.
3. Initialize training parameters.
4. Execute training episodes.
5. Save statistics.
6. Save checkpoints.
7. Update exploration.
8. Synchronize the target network.

---

## Important Variables

* environment
* agent
* epsilon
* episode reward
* replay memory
* loss

---

## Output

Produces:

* trained checkpoint,
* training CSV,
* training statistics.

---

# 7.3 `play.py`

## Purpose

Runs a trained agent without further learning.

Unlike `train.py`, this program never updates the neural network.

Instead, it evaluates the policy learned during training.

---

## Main Workflow

1. Load checkpoint.
2. Create environment.
3. Initialize frame stack.
4. Select greedy actions.
5. Capture gameplay frames.
6. Generate video.
7. Save evaluation summary.

---

## Outputs

Produces:

* gameplay frames,
* MP4 video,
* JSON summary.

---

# 7.4 `plot_training.py`

## Purpose

Reads the training log and produces visualizations.

The program converts the raw CSV data into informative learning curves.

---

## Generated Figures

* Reward vs Episode
* Loss vs Episode
* Epsilon vs Episode

These plots provide a concise overview of training progress.

---

# 7.5 `agent.py`

## Purpose

The `MarioAgent` class represents the reinforcement learning agent.

This is the central component of the project.

Nearly all reinforcement learning functionality is encapsulated within this class.

---

## Main Responsibilities

* action selection,
* experience storage,
* learning,
* checkpoint management,
* target network synchronization.

---

## Main Attributes

### Neural Networks

* online network
* target network

---

### Learning Components

* optimizer
* loss function

---

### Memory

* replay memory

---

### Device

Automatically selects:

* CUDA (GPU)
* CPU

depending on availability.

---

## Main Methods

### `select_action()`

Uses the ε-greedy strategy.

Returns the selected action.

---

### `remember()`

Stores an experience inside replay memory.

---

### `learn()`

Performs one optimization step.

Main operations:

* sample mini-batch,
* compute current Q-values,
* compute target Q-values,
* calculate Bellman loss,
* backpropagation,
* optimizer update.

---

### `update_target_model()`

Copies the online network into the target network.

This operation is performed periodically during training.

---

### `save()`

Stores the learned network parameters.

---

### `load()`

Loads a previously trained checkpoint.

---

# 7.6 `model.py`

## Purpose

Defines the Deep Q-Network architecture.

The network receives the stacked game frames and predicts one Q-value for every possible action.

---

## Architecture

The model consists of:

* convolutional layers,
* activation functions,
* fully connected layers,
* output layer.

---

## Input

```text id="bwzcr5"
(4, 84, 84)
```

---

## Output

```text id="yqv4a2"
Number of actions
```

Each output neuron corresponds to one possible action.

---

# 7.7 `memory.py`

## Purpose

Implements Replay Memory.

Replay Memory stores previous experiences for later reuse.

---

## Experience Format

Each experience contains:

* state,
* action,
* reward,
* next state,
* done flag.

---

## Main Methods

### `push()`

Stores a new experience.

---

### `sample()`

Returns a random mini-batch.

---

### `__len__()`

Returns the current memory size.

---

# 7.8 `wrappers.py`

## Purpose

Contains helper functions related to the game environment.

These functions simplify preprocessing and state management.

---

## Main Functions

### `preprocess_frame()`

Converts an RGB frame into the processed grayscale image used by the neural network.

---

### `create_initial_stack()`

Creates the first stack of frames after environment reset.

---

### `update_stack()`

Maintains the sliding window of the most recent four frames.

---

### `step_with_frame_skip()`

Executes the selected action for multiple consecutive frames.

Frame skipping:

* reduces computation,
* accelerates learning,
* smooths gameplay.

---

# 7.9 `utils.py`

## Purpose

Reserved for future helper functions.

Although only lightly used in Version 1.0, separating utility functions into their own module keeps the project organized as it grows.

---

# 7.10 `__init__.py`

## Purpose

Marks `mario_agent` as a Python package.

This enables imports such as:

```python
from mario_agent.agent import MarioAgent
```

without exposing implementation details.

---

# 7.11 Relationships Between Files

The following diagram summarizes the interactions between the major modules.

```text id="5r5vaf"
train.py
    │
    ▼
MarioAgent
    │
    ├──────────► MarioDQN
    │
    ├──────────► ReplayMemory
    │
    └──────────► Wrappers
                    │
                    ▼
            Mario Environment
```

Evaluation follows a similar structure.

```text id="ymzt1q"
play.py
    │
    ▼
MarioAgent
    │
    ▼
Loaded Checkpoint
    │
    ▼
Gameplay Recording
```

---

# Summary

Although the project consists of only a modest number of source files, each one has a clearly defined responsibility. This separation of concerns makes the codebase easier to understand, test, and maintain.

The next chapter focuses on the project's main classes and functions in greater detail, explaining how the individual components collaborate during training and evaluation.

# 8. Main Classes and Functions

The previous chapter introduced the purpose of each source file. This chapter examines the project's principal classes and functions in greater detail, explaining their responsibilities and how they collaborate during training and evaluation.

The implementation intentionally follows an object-oriented design in which each class encapsulates a specific aspect of the reinforcement learning system.

---

# 8.1 The `MarioAgent` Class

The `MarioAgent` class is the central controller of the project.

Nearly all reinforcement learning logic is encapsulated within this class.

Rather than allowing the training script to directly manipulate neural networks, replay memory, optimizers, and checkpoints, the training program interacts almost exclusively with the `MarioAgent`.

This design significantly simplifies the remainder of the codebase.

---

## Responsibilities

The agent is responsible for:

* selecting actions,
* storing experiences,
* learning from replay memory,
* updating the neural network,
* maintaining the target network,
* saving checkpoints,
* loading checkpoints.

The training script therefore becomes a high-level coordinator rather than containing reinforcement learning logic itself.

---

## Main Attributes

### Online Network

The online network estimates the Q-values used to select actions and is updated continuously during training.

---

### Target Network

The target network provides stable target values during Bellman updates.

It is synchronized periodically with the online network.

---

### Replay Memory

Stores previous experiences for random sampling.

---

### Optimizer

Responsible for updating the neural network parameters during backpropagation.

This project uses the Adam optimizer.

---

### Loss Function

Measures the difference between predicted Q-values and Bellman target values.

Version 1.0 uses the Huber Loss (`SmoothL1Loss`).

---

### Device

Automatically selects either:

* CUDA
* CPU

depending on hardware availability.

---

# 8.2 Main Methods of `MarioAgent`

---

## `select_action()`

### Purpose

Chooses the next action according to the ε-greedy policy.

---

### Inputs

* current state
* epsilon

---

### Processing

If a randomly generated number is less than ε:

* choose a random action.

Otherwise:

* run the neural network,
* compute Q-values,
* select the highest-valued action.

---

### Output

Returns the selected action index.

---

## `remember()`

### Purpose

Stores one experience in replay memory.

---

### Inputs

* current state
* action
* reward
* next state
* done flag

---

### Output

None.

The experience becomes available for future learning.

---

## `can_learn()`

### Purpose

Determines whether replay memory contains enough experiences to form a mini-batch.

---

### Input

Batch size.

---

### Output

Boolean value.

---

## `learn()`

### Purpose

Performs one reinforcement learning update.

This is the most important method in the entire project.

---

### Internal Workflow

1. Verify enough replay memory exists.
2. Sample a random mini-batch.
3. Convert data into tensors.
4. Predict current Q-values.
5. Predict next-state Q-values using the target network.
6. Compute Bellman targets.
7. Compute Huber loss.
8. Perform backpropagation.
9. Update network parameters.

---

### Output

Returns the current training loss.

---

## `update_target_model()`

### Purpose

Copies the weights of the online network into the target network.

This operation stabilizes learning by preventing the target values from changing after every optimization step.

---

## `save()`

Stores the current neural network parameters.

---

## `load()`

Loads a previously trained checkpoint.

---

# 8.3 The `MarioDQN` Class

The `MarioDQN` class defines the neural network architecture.

Its sole responsibility is to map a game state into predicted Q-values.

---

## Input

Stacked game frames.

```text
(4, 84, 84)
```

---

## Output

One Q-value for every available action.

Example:

```text
Action 0 → 0.84
Action 1 → 2.15
Action 2 → 1.33
...
```

The network itself does not decide which action to execute.

It simply estimates expected future rewards.

---

## Forward Pass

The `forward()` method performs:

1. convolution,
2. feature extraction,
3. flattening,
4. fully connected layers,
5. Q-value prediction.

---

# 8.4 The `ReplayMemory` Class

Replay Memory is responsible for storing experiences collected during gameplay.

---

## Responsibilities

* add new experiences,
* remove old experiences when capacity is reached,
* randomly sample mini-batches.

---

## Stored Experience

Each experience consists of:

```text
(state,
 action,
 reward,
 next_state,
 done)
```

---

## Main Methods

### `push()`

Stores one experience.

---

### `sample()`

Returns a random mini-batch.

---

### `__len__()`

Returns the current memory size.

---

# 8.5 Helper Functions in `wrappers.py`

Unlike the previous classes, `wrappers.py` contains standalone functions.

These functions simplify interaction with the environment.

---

## `preprocess_frame()`

### Purpose

Transforms the raw RGB image into the neural network input.

Processing steps include:

* grayscale conversion,
* resizing,
* normalization.

---

## `create_initial_stack()`

Creates the first stack after resetting the environment.

Initially, the same processed frame is repeated to create a stack of four frames.

---

## `update_stack()`

Maintains a sliding window of the four most recent frames.

The oldest frame is discarded whenever a new frame is received.

---

## `step_with_frame_skip()`

Executes one chosen action over several consecutive game frames.

Frame skipping reduces computation while preserving meaningful gameplay.

---

# 8.6 Relationship Between Classes

The following diagram illustrates the interaction between the principal classes.

```text
              MarioAgent
                   │
     ┌─────────────┼──────────────┐
     │             │              │
     ▼             ▼              ▼
MarioDQN     ReplayMemory    Wrappers
     │
     ▼
Q-values
```

The `MarioAgent` coordinates all other components.

Each supporting class has a narrow, clearly defined responsibility.

---

# 8.7 Object-Oriented Design Benefits

Encapsulating functionality into classes provides several advantages.

### Readability

Each class has a single responsibility.

---

### Reusability

Replay Memory and the neural network could easily be reused in another reinforcement learning project.

---

### Testability

Each class can be validated independently before integration.

---

### Maintainability

Changes to one component rarely require modifications elsewhere.

---

### Extensibility

Additional reinforcement learning algorithms can be incorporated without restructuring the project.

---

# Summary

The project uses only a small number of classes, but each plays an essential role.

The `MarioAgent` orchestrates the reinforcement learning process, while the remaining classes provide focused functionality for neural network inference, experience storage, preprocessing, and interaction with the game environment.

The next chapter explains how the project's behavior can be modified through the centralized configuration file (`config.py`) and how different experiments can be performed simply by adjusting configuration parameters.

---

# 9. Configuration Management (`config.py`)

One of the guiding software engineering principles of this project is that **all tunable parameters should be centralized in a single location**.

Rather than embedding numerical constants throughout the source code, the project stores nearly all configurable values in `config.py`. This approach improves readability, simplifies experimentation, and helps ensure that the entire project remains internally consistent.

Changing a parameter in `config.py` automatically affects every module that imports it.

---

# 9.1 Why Centralize Configuration?

During development, numerous parameters needed to be adjusted repeatedly, including:

* image size,
* frame skip,
* learning rate,
* discount factor,
* batch size,
* epsilon schedule,
* number of training episodes,
* video generation settings.

If these values were duplicated throughout the codebase, experiments would become difficult to manage and inconsistent configurations could easily occur.

Centralizing configuration provides several benefits:

* Single source of truth
* Easier experimentation
* Improved reproducibility
* Cleaner source code
* Reduced maintenance effort

---

# 9.2 Environment Parameters

These parameters determine how observations are generated before being passed to the neural network.

| Parameter    | Purpose                                                        |
| ------------ | -------------------------------------------------------------- |
| `FRAME_SKIP` | Number of consecutive frames executed for each selected action |
| `STACK_SIZE` | Number of consecutive processed frames forming one state       |
| `IMAGE_SIZE` | Width and height of processed images                           |

### Effect of Changing These Values

**FRAME_SKIP**

Increasing the value:

* reduces computation,
* speeds up training,
* may reduce control precision.

Decreasing the value:

* improves control,
* increases computational cost.

---

**STACK_SIZE**

Larger stacks provide more temporal information but increase memory usage.

Smaller stacks reduce computation but may lose important motion information.

---

**IMAGE_SIZE**

Higher resolution:

* preserves more visual detail,
* increases computation.

Lower resolution:

* accelerates learning,
* may remove useful information.

---

# 9.3 Learning Parameters

These parameters influence the learning algorithm itself.

| Parameter       | Purpose             |
| --------------- | ------------------- |
| `GAMMA`         | Discount factor     |
| `LEARNING_RATE` | Optimizer step size |
| `BATCH_SIZE`    | Mini-batch size     |

---

## Discount Factor (`GAMMA`)

The discount factor determines how much importance is assigned to future rewards.

Values close to:

```text id="4lqkv1"
1.0
```

encourage long-term planning.

Smaller values place greater emphasis on immediate rewards.

---

## Learning Rate

Controls how rapidly the neural network parameters are updated.

Very small values:

* slow learning.

Very large values:

* may destabilize training.

---

## Batch Size

Determines how many experiences are sampled from replay memory during each learning step.

Larger mini-batches generally produce more stable gradient estimates but require more memory.

---

# 9.4 Training Parameters

These parameters control the overall training process.

| Parameter      | Purpose                           |
| -------------- | --------------------------------- |
| `NUM_EPISODES` | Number of training episodes       |
| `MAX_STEPS`    | Maximum steps allowed per episode |

---

## Number of Episodes

Increasing the number of episodes generally provides more learning opportunities.

However, longer training also increases execution time.

---

## Maximum Steps

Limits the duration of an individual episode.

Episodes may terminate earlier if Mario dies or completes the level.

---

# 9.5 Exploration Parameters

Exploration is controlled through the ε-greedy strategy.

| Parameter       | Purpose                                         |
| --------------- | ----------------------------------------------- |
| `EPSILON_START` | Initial exploration rate                        |
| `EPSILON_MIN`   | Minimum exploration rate                        |
| `EPSILON_DECAY` | Multiplicative decay applied after each episode |

---

## Exploration Schedule

Training begins with:

```text id="pt07eg"
ε = EPSILON_START
```

meaning that nearly all actions are selected randomly.

After each episode:

```text id="i2n6u8"
ε ← max(EPSILON_MIN,
        ε × EPSILON_DECAY)
```

As ε decreases, the agent gradually shifts from exploration toward exploitation.

---

# 9.6 Target Network Parameters

The target network is synchronized periodically.

| Parameter             | Purpose                                           |
| --------------------- | ------------------------------------------------- |
| `TARGET_UPDATE_EVERY` | Number of episodes between target network updates |

More frequent synchronization allows the target network to track the online network closely.

Less frequent synchronization generally improves training stability.

---

# 9.7 Logging Parameters

These parameters control experiment tracking.

| Parameter           | Purpose                               |
| ------------------- | ------------------------------------- |
| `LOGS_DIR`          | Directory containing training outputs |
| `TRAINING_LOG_FILE` | CSV training log                      |

The training log stores:

* reward,
* loss,
* epsilon,
* replay memory size,
* episode statistics.

These data are later used to generate training plots.

---

# 9.8 Evaluation Parameters

Evaluation produces gameplay recordings.

| Parameter            | Purpose                              |
| -------------------- | ------------------------------------ |
| `SAVE_EVERY_N_STEPS` | Interval between captured frames     |
| `VIDEO_FPS`          | Frames per second of generated video |

Reducing the frame-saving frequency decreases storage requirements.

Increasing the frame rate produces smoother videos but larger output files.

---

# 9.9 Checkpoint Parameters

| Parameter         | Purpose                                |
| ----------------- | -------------------------------------- |
| `CHECKPOINT_FILE` | Default filename for the trained model |

Using a centralized checkpoint name ensures that both `train.py` and `play.py` always refer to the same model.

Changing the filename requires modification in only one location.

---

# 9.10 Typical Experiment

A complete experiment might proceed as follows:

1. Modify selected parameters in `config.py`.
2. Train the model.
3. Generate learning plots.
4. Evaluate the trained agent.
5. Compare the results with previous experiments.

Because every configurable value is centralized, repeating experiments with different settings becomes straightforward.

---

# 9.11 Configuration as Documentation

An additional advantage of `config.py` is that it documents the assumptions underlying the project.

Instead of searching through multiple source files, readers can quickly determine:

* image dimensions,
* learning rate,
* replay settings,
* exploration schedule,
* evaluation settings,
* logging behavior.

The configuration file therefore serves as both executable code and project documentation.

---

# 9.12 Best Practices

During development, several configuration management practices proved particularly useful:

* Keep all tunable parameters in one file.
* Avoid hard-coded constants elsewhere in the project.
* Use descriptive parameter names.
* Group related parameters into logical sections.
* Add comments where appropriate.
* Record configuration values alongside experimental results.

These practices make experiments easier to reproduce and simplify future extensions.

---

# Summary

The introduction of `config.py` represented an important improvement in the project's architecture.

By separating configuration from implementation, the project became cleaner, easier to maintain, and significantly more flexible. New experiments can now be performed simply by modifying a small number of configuration values rather than editing multiple source files.

The next chapter describes how each subsystem was tested independently before integration and explains the role of the project's test programs.

---

# 10. Testing Individual Components

A distinguishing feature of this project is that every major component was tested independently before being integrated into the complete reinforcement learning system.

Rather than attempting to debug an entire training pipeline at once, each subsystem was verified in isolation. This incremental testing strategy greatly simplified development and made it easier to identify and correct implementation errors.

The project therefore includes several standalone test programs, each focusing on a specific subsystem.

```text
test_env.py
test_model.py
test_memory.py
test_agent.py
```

Together, these tests provided confidence that the individual building blocks functioned correctly before the complete agent was trained.

---

# 10.1 Why Test Components Independently?

Reinforcement learning systems combine several interacting components:

* game environment,
* image preprocessing,
* neural network,
* replay memory,
* optimization,
* action selection,
* learning algorithm.

If all components are developed simultaneously, locating the source of an error becomes extremely difficult.

Instead, this project adopted the following workflow:

```text
Implement
      ↓
Test
      ↓
Verify
      ↓
Integrate
      ↓
Repeat
```

Only after one component had been validated was the next component introduced.

This approach proved invaluable throughout development.

---

# 10.2 `test_env.py`

## Purpose

Verifies that the game environment and preprocessing pipeline function correctly.

---

## What is Tested?

### Environment Creation

Confirms that the Super Mario Bros environment can be created successfully.

---

### Observation Dimensions

Verifies that observations have the expected dimensions.

Example:

```text
(240, 256, 3)
```

---

### Action Space

Displays the available action space.

Example:

```text
Discrete(7)
```

---

### Image Preprocessing

Tests:

* grayscale conversion,
* image resizing,
* normalization.

---

### Frame Stacking

Confirms that stacked states have the expected dimensions.

Example:

```text
(4, 84, 84)
```

---

### Frame Skipping

Verifies that actions can be executed over multiple frames and that the environment returns:

* next observation,
* reward,
* done flag,
* information dictionary.

---

## Why It Matters

Without a functioning environment, no reinforcement learning can occur.

Testing the environment first ensured that all subsequent development was built on a reliable foundation.

---

# 10.3 `test_model.py`

## Purpose

Validates the Deep Q-Network implementation.

---

## What is Tested?

### Model Construction

Confirms that the neural network can be instantiated successfully.

---

### Forward Pass

Creates dummy input tensors and passes them through the network.

---

### Output Dimensions

Verifies that the output vector contains one Q-value per available action.

Example:

```text
Input:
(1, 4, 84, 84)

Output:
(1, 7)
```

---

## Why It Matters

Many reinforcement learning errors originate from tensor dimension mismatches.

Testing the model independently eliminates this entire class of problems before integration.

---

# 10.4 `test_memory.py`

## Purpose

Validates the Replay Memory implementation.

---

## What is Tested?

### Inserting Experiences

Confirms that new experiences are stored correctly.

---

### Memory Size

Verifies that the reported memory length increases as expected.

---

### Random Sampling

Samples random mini-batches from replay memory.

---

### Returned Structure

Confirms that each sampled experience contains:

* state,
* action,
* reward,
* next state,
* done flag.

---

## Why It Matters

Replay Memory is fundamental to stable DQN training.

Testing it independently ensured that experiences were being stored and retrieved correctly before learning began.

---

# 10.5 `test_agent.py`

## Purpose

Validates the complete `MarioAgent` class.

This is the most comprehensive of the individual tests.

---

## What is Tested?

### Agent Construction

Creates a complete reinforcement learning agent.

---

### Device Selection

Verifies automatic selection of:

* CUDA,
* or CPU.

---

### Action Selection

Tests both:

* greedy action selection,
* random exploration.

---

### Replay Memory

Stores sample experiences.

Verifies:

* memory size,
* replay insertion.

---

### Learning

Exercises the complete learning pipeline.

Including:

* mini-batch sampling,
* Q-value prediction,
* Bellman target calculation,
* loss computation,
* backpropagation,
* optimizer update.

---

### Checkpoint Management

Verifies:

* saving,
* loading,
* model restoration.

---

## Why It Matters

By the time `test_agent.py` passed successfully, nearly every subsystem of the reinforcement learning implementation had already been validated.

Consequently, debugging the full training loop became considerably easier.

---

# 10.6 Integration Testing

After the individual components had been validated, the complete project was tested using three executable programs.

---

## Training

```bash
python train.py
```

Verifies:

* environment interaction,
* continuous learning,
* replay memory,
* target network updates,
* checkpoint generation,
* CSV logging.

---

## Plot Generation

```bash
python plot_training.py
```

Verifies:

* CSV loading,
* reward plot,
* loss plot,
* epsilon plot.

---

## Evaluation

```bash
python play.py
```

Verifies:

* checkpoint loading,
* gameplay execution,
* frame capture,
* video generation,
* JSON summary generation.

---

# 10.7 Final Validation

A successful Version 1.0 execution produced:

```text
mario_dqn.pth

logs/
    training.csv
    reward_vs_episode.png
    loss_vs_episode.png
    epsilon_vs_episode.png

runs/
    run_0001/
        frames/
        run.mp4
        summary.json
```

The successful generation of these artifacts confirmed that the complete reinforcement learning pipeline was functioning correctly.

---

# 10.8 Lessons Learned from Testing

Several important software engineering lessons emerged during development.

### Test Early

Small problems are much easier to diagnose before additional complexity is introduced.

---

### Test Often

Frequent testing catches regressions immediately after they occur.

---

### Test Components Independently

Each module should be validated before integration.

---

### Verify Intermediate Results

Checking tensor dimensions, replay memory contents, and preprocessing outputs often reveals errors long before training begins.

---

### Build Incrementally

Developing one subsystem at a time proved to be significantly more manageable than attempting to construct the entire reinforcement learning system in a single step.

---

# Summary

Independent component testing was one of the most valuable development practices used throughout this project.

By validating each subsystem individually before integration, the implementation remained easier to debug, easier to understand, and more reliable.

The next chapter brings all of these components together by describing the complete workflow for training, evaluating, and documenting reinforcement learning experiments.

---

# 11. Running the Complete Project

Once all components have been implemented and individually tested, the complete reinforcement learning workflow consists of three sequential stages:

1. **Training the agent**
2. **Generating training visualizations**
3. **Evaluating the trained agent**

This chapter describes the complete workflow from an empty project directory to a fully trained model together with all generated artifacts.

---

# 11.1 Typical Workflow

The complete workflow can be summarized as:

```text
        train.py
            │
            ▼
   mario_dqn.pth
            │
            ▼
    training.csv
            │
            ▼
 plot_training.py
            │
            ▼
   Learning Curves
            │
            ▼
        play.py
            │
            ▼
 Frames • Video • Summary
```

Each stage builds upon the outputs generated by the previous stage.

---

# 11.2 Step 1 – Configure the Experiment

Before training begins, review the values in:

```text
config.py
```

Typical parameters include:

* number of episodes,
* maximum steps,
* learning rate,
* replay batch size,
* epsilon schedule,
* target network update frequency.

For example:

```python
NUM_EPISODES = 100
MAX_STEPS = 1000

EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.95
```

Changing these values allows different experiments to be performed without modifying the implementation.

---

# 11.3 Step 2 – Train the Agent

Execute:

```bash
python train.py
```

During training, the following operations occur repeatedly:

* environment reset,
* state preprocessing,
* frame stacking,
* action selection,
* environment interaction,
* experience storage,
* mini-batch sampling,
* Bellman update,
* network optimization,
* replay memory growth,
* target network synchronization,
* checkpoint saving.

Example console output:

```text
Episode 42
Steps: 127
Episode reward: 642
Memory length: 4218
Last loss: 3.74
Epsilon: 0.29
```

---

# 11.4 Outputs Produced During Training

At the end of training, several files have been generated automatically.

### Model Checkpoint

```text
mario_dqn.pth
```

Contains the trained neural network parameters.

---

### Training Log

```text
logs/training.csv
```

Contains one row per episode.

Typical information includes:

* episode,
* reward,
* loss,
* epsilon,
* replay memory size.

---

# 11.5 Step 3 – Generate Learning Curves

Execute:

```bash
python plot_training.py
```

The program reads:

```text
training.csv
```

and generates:

```text
reward_vs_episode.png

loss_vs_episode.png

epsilon_vs_episode.png
```

These plots provide a concise overview of the learning process.

---

# 11.6 Interpreting the Plots

### Reward Curve

An increasing reward generally indicates improved gameplay.

Large fluctuations are common during reinforcement learning.

---

### Loss Curve

The loss should remain reasonably stable.

Occasional spikes are expected because the training data continuously changes.

---

### Epsilon Curve

The epsilon curve should show gradual decay from:

```text
1.0
```

toward the configured minimum value.

This reflects the transition from exploration to exploitation.

---

# 11.7 Step 4 – Evaluate the Agent

Execute:

```bash
python play.py
```

Unlike training, evaluation performs **no learning**.

Instead, the agent:

* loads the trained checkpoint,
* disables exploration,
* selects greedy actions,
* records gameplay.

---

# 11.8 Evaluation Outputs

Evaluation automatically creates a new run directory.

Example:

```text
runs/
└── run_0001/
```

This directory contains:

```text
frames/

run.mp4

summary.json
```

---

## Frames

Selected gameplay frames captured during evaluation.

Each frame contains an overlay showing:

* current step,
* accumulated reward,
* selected action,
* Mario's horizontal position.

---

## Video

The saved MP4 file provides a qualitative assessment of the learned policy.

Unlike numerical metrics, the video allows direct observation of the agent's behavior.

---

## Summary File

The JSON summary records:

* checkpoint used,
* reward,
* number of steps,
* final X position,
* timestamp,
* evaluation settings.

This metadata supports reproducibility and experiment comparison.

---

# 11.9 Complete Output Directory

After a successful experiment, the project directory contains:

```text
super_mario_learning/

│
├── mario_dqn.pth
│
├── logs/
│   ├── training.csv
│   ├── reward_vs_episode.png
│   ├── loss_vs_episode.png
│   └── epsilon_vs_episode.png
│
└── runs/
    └── run_0001/
        ├── frames/
        ├── run.mp4
        └── summary.json
```

These artifacts collectively document both the quantitative and qualitative performance of the trained agent.

---

# 11.10 Typical Experiment Lifecycle

A complete experiment generally follows this sequence:

```text
Modify config.py
        │
        ▼
Train
        │
        ▼
Analyze Plots
        │
        ▼
Evaluate Agent
        │
        ▼
Review Video
        │
        ▼
Adjust Parameters
        │
        ▼
Repeat
```

This iterative process is typical of reinforcement learning research and development.

---

# 11.11 Reproducibility

One of the primary goals of the project is reproducibility.

Every experiment automatically generates:

* configuration settings,
* model checkpoint,
* training log,
* learning curves,
* gameplay recording,
* evaluation metadata.

These outputs make it possible to revisit and compare experiments without rerunning the entire training process.

---

# 11.12 Version 1.0 Workflow

The completed Version 1.0 implementation supports the entire reinforcement learning pipeline:

* configurable experiments,
* automated training,
* checkpoint generation,
* experiment logging,
* visualization,
* evaluation,
* gameplay recording,
* metadata generation.

This represents a complete end-to-end Deep Q-Network implementation suitable for further experimentation and future extensions.

---

# Summary

The project workflow follows a clear sequence:

1. Configure the experiment.
2. Train the reinforcement learning agent.
3. Generate learning curves.
4. Evaluate the trained policy.
5. Review the resulting artifacts.

The following chapter summarizes the outputs generated by the project and discusses the principal lessons learned during development, together with possible directions for future enhancements.

---

# 12. Project Outputs, Lessons Learned, and Future Enhancements

This chapter concludes the documentation by summarizing the outputs generated by the project, reflecting on the development process, and identifying opportunities for future improvements.

Although Version 1.0 represents a relatively compact reinforcement learning project, it incorporates many of the architectural principles found in larger reinforcement learning systems.

---

# 12.1 Project Outputs

Every successful training and evaluation cycle produces a collection of artifacts documenting both the learning process and the final agent.

These artifacts can be broadly categorized into four groups.

---

## Model Artifacts

### Trained Model Checkpoint

```text id="byqzyu"
mario_dqn.pth
```

Contains the learned neural network parameters.

This checkpoint can be:

* loaded for evaluation,
* used to resume training,
* archived for comparison with future experiments.

---

## Training Artifacts

Located in:

```text id="u9mnh7"
logs/
```

Typical contents include:

```text id="4m6kj2"
training.csv

reward_vs_episode.png

loss_vs_episode.png

epsilon_vs_episode.png
```

These files provide a quantitative record of the training process.

---

## Evaluation Artifacts

Located in:

```text id="j42mzd"
runs/run_xxxx/
```

Typical contents include:

```text id="0gnhp0"
frames/

run.mp4

summary.json
```

These outputs provide a qualitative assessment of the trained policy.

---

## Configuration

The configuration file itself also becomes an important artifact.

```text id="9qndv6"
config.py
```

Together with the checkpoint and logs, it enables experiments to be reproduced accurately.

---

# 12.2 Lessons Learned

Developing this project provided valuable insights into both reinforcement learning and software engineering.

Several observations proved particularly important.

---

## Reinforcement Learning Is Incremental

Unlike supervised learning, reinforcement learning agents typically exhibit highly variable behavior during training.

Progress is rarely linear.

Periods of apparent improvement may be followed by temporary regressions before the agent ultimately converges toward a better policy.

Understanding this behavior helps set realistic expectations during experimentation.

---

## Good State Representation Matters

The quality of the input representation strongly influences learning performance.

Frame preprocessing and frame stacking proved essential for enabling the neural network to infer motion and temporal relationships.

---

## Replay Memory Improves Stability

Randomly sampling experiences from replay memory significantly reduced correlations between training examples.

Without replay memory, learning became considerably less stable.

---

## Target Networks Are Essential

Separating the online network from the target network stabilized Bellman updates and reduced oscillations during training.

This relatively small architectural change had a substantial impact on learning stability.

---

## Exploration Must Be Controlled

Beginning training with a high exploration rate encouraged the agent to discover diverse behaviors.

Gradually reducing exploration allowed the learned policy to become increasingly deterministic.

Balancing exploration and exploitation proved to be one of the central challenges of reinforcement learning.

---

# 12.3 Software Engineering Lessons

Beyond reinforcement learning, the project reinforced several important software engineering practices.

---

## Build Incrementally

Developing one subsystem at a time dramatically simplified debugging.

Rather than constructing the entire reinforcement learning pipeline at once, each component was validated independently before integration.

This strategy reduced both development time and implementation errors.

---

## Modular Design Simplifies Maintenance

Separating preprocessing, learning, replay memory, evaluation, visualization, and configuration into independent modules resulted in a cleaner and more maintainable codebase.

Each module has a clearly defined responsibility.

---

## Centralized Configuration Is Valuable

Moving all configurable parameters into `config.py` greatly improved the project's flexibility.

Experiments can now be performed simply by modifying configuration values rather than editing multiple source files.

---

## Test Components Independently

Dedicated test programs for each subsystem significantly reduced debugging effort.

By the time the complete training pipeline was assembled, most implementation errors had already been eliminated.

---

## Record Experimental Results

Automatically generating logs, plots, checkpoints, videos, and metadata makes experiments easier to analyze and reproduce.

Experiment tracking should be considered an integral part of any machine learning project.

---

# 12.4 Challenges Encountered

Several practical challenges arose during development.

Examples included:

* package version compatibility,
* CUDA configuration,
* Gym deprecation warnings,
* NumPy and OpenCV interactions,
* replay memory debugging,
* tensor dimension mismatches,
* Bellman target implementation,
* target network synchronization,
* experiment management.

Addressing these issues contributed significantly to understanding both the reinforcement learning algorithm and the supporting software ecosystem.

---

# 12.5 Possible Future Enhancements

Version 1.0 establishes a solid foundation that can be extended in several directions.

Potential improvements include:

### Reinforcement Learning Algorithms

* Double DQN
* Dueling DQN
* Prioritized Experience Replay
* Multi-step learning

---

### Training Improvements

* Longer training runs
* Improved reward shaping
* Automatic checkpoint management
* Early stopping criteria

---

### Visualization

* Live training dashboards
* TensorBoard integration
* Interactive experiment comparison

---

### Evaluation

* Multiple evaluation episodes
* Statistical performance summaries
* Comparative checkpoint evaluation

---

### Software Engineering

* Automated unit testing
* Configuration validation
* Command-line argument support
* Automated experiment naming
* Packaging for distribution

---

# 12.6 Final Remarks

This project demonstrates that a complete reinforcement learning system can be built through a sequence of manageable, well-defined steps.

Starting from a simple game environment, the implementation gradually evolved into a modular Deep Q-Network capable of learning from interaction, recording experimental results, and evaluating its learned policy.

Equally important, the project emphasizes that successful reinforcement learning depends not only on the underlying algorithm but also on careful software design, systematic testing, reproducible experimentation, and disciplined project organization.

The resulting codebase provides a strong foundation for future exploration of more advanced reinforcement learning methods while remaining compact enough to be understood in its entirety.

---

# Acknowledgements

This project was developed as a learning exercise to gain practical experience with Deep Reinforcement Learning using PyTorch and the Super Mario Bros environment.

The implementation draws upon well-established ideas from the Deep Q-Network (DQN) literature while emphasizing incremental development, modular software design, and reproducible experimentation.

---

# Conclusion

Version 1.0 successfully achieves its original objectives:

* A complete Deep Q-Network implementation.
* A modular and maintainable software architecture.
* A reproducible experimentation workflow.
* Independent testing of all major components.
* Automated training, evaluation, visualization, and documentation.

Most importantly, the project transformed reinforcement learning from an abstract concept into a practical, working system whose individual components—and the interactions between them—are fully understood and documented.

---

# 13. Lessons Learned

Developing this project provided valuable insights into both reinforcement learning and software engineering. While many of these lessons are well known in theory, implementing the system from scratch highlighted their practical importance.

---

## 13.1 Environment Compatibility Is Critical

One of the earliest lessons learned was that reinforcement learning projects often depend on a complex ecosystem of libraries whose versions must be compatible.

During development, several issues were encountered involving:

- Gym versus Gymnasium
- Gym Super Mario Bros
- NES-Py
- NumPy
- OpenCV
- CUDA-enabled PyTorch

In particular, changing the NumPy version to satisfy one package introduced compatibility problems elsewhere. Maintaining a stable Conda environment with carefully selected package versions proved essential.

**Lesson learned:** Once a working environment has been established, avoid unnecessary package upgrades. Record the package versions so the environment can be reproduced later.

---

## 13.2 Plan Configuration Early

Initially, many hyperparameters were defined directly inside the source files.

As the project grew, changing values such as the learning rate, epsilon schedule, frame skip, or image size required editing several files.

Introducing a centralized `config.py` significantly improved the project's organization.

Benefits included:

- easier experimentation,
- cleaner source code,
- improved consistency,
- simplified maintenance.

**Lesson learned:** Centralize configuration at the beginning of a project whenever possible.

---

## 13.3 Build Incrementally

Rather than implementing the entire reinforcement learning pipeline at once, the project was developed one component at a time.

The sequence was:

1. Environment
2. Frame preprocessing
3. Frame stacking
4. Neural network
5. Replay memory
6. Agent
7. Training
8. Evaluation
9. Visualization

Each component was tested before moving to the next.

This incremental approach dramatically simplified debugging.

**Lesson learned:** Validate one component before introducing another.

---

## 13.4 Test Components Independently

Dedicated test programs were created for:

- the environment,
- the neural network,
- replay memory,
- the reinforcement learning agent.

When integration began, most implementation errors had already been eliminated.

**Lesson learned:** Independent component testing reduces debugging time and increases confidence in the final system.

---

## 13.5 State Representation Matters

Initially, it might seem sufficient to feed raw game images directly into the neural network.

However, preprocessing and frame stacking proved essential.

Converting frames to grayscale, resizing them, and stacking four consecutive observations produced a compact yet informative state representation capable of capturing motion.

**Lesson learned:** A carefully designed state representation often contributes more to learning performance than simply increasing model complexity.

---

## 13.6 Two Networks Are Better Than One

One of the most important architectural decisions in the DQN algorithm is the use of separate online and target networks.

Although maintaining two neural networks initially appears redundant, the target network significantly improves training stability by preventing the learning target from changing continuously.

Periodic synchronization produced noticeably more stable learning.

**Lesson learned:** The target network is a key ingredient of stable Deep Q-Network training.

---

## 13.7 Replay Memory Improves Learning

Training directly on consecutive game frames produced highly correlated experiences.

Replay memory solved this problem by randomizing the training data.

This led to more stable optimization and better use of collected experiences.

**Lesson learned:** Replay memory is fundamental to effective Deep Q-Network training.

---

## 13.8 Hyperparameters Require Experimentation

Training performance depended heavily on parameters such as:

- learning rate,
- batch size,
- epsilon decay,
- discount factor,
- target network update frequency.

There is no universally optimal set of values.

Effective reinforcement learning requires experimentation and careful observation of training results.

**Lesson learned:** Hyperparameter tuning is an essential part of reinforcement learning.

---

## 13.9 Experiment Tracking Is Worth the Effort

Automatically saving:

- checkpoints,
- CSV logs,
- learning curves,
- gameplay videos,
- evaluation summaries,

made it much easier to understand the behavior of the agent and compare different training runs.

**Lesson learned:** Good experiment tracking should be considered part of the learning algorithm rather than an optional addition.

---

## 13.10 Good Software Engineering Matters

Although the focus of the project was reinforcement learning, many of the most valuable lessons concerned software engineering.

Modular design, centralized configuration, incremental development, testing, and reproducibility all contributed significantly to the success of the project.

These practices will remain valuable regardless of the specific machine learning algorithm being implemented.

---

I would make Section 14 the **umbrella chapter** that introduces the roadmap, then divide it into two subsections:

* **14.1 Software & Usability Enhancements**
* **14.2 Experimentation & Analysis Enhancements**
* **14.3 Advanced Reinforcement Learning Enhancements** (the section we already wrote)

This creates a logical progression from improving the tooling, to improving experimentation, to improving the learning algorithm itself.

# 14. Future Enhancements

Version 1.0 provides a complete, modular implementation of a Deep Q-Network (DQN) agent capable of learning to play the original Super Mario Bros game. While the project successfully achieves its primary objectives, it also establishes a solid foundation for future development.

The modular architecture adopted throughout the project makes it possible to extend individual components without requiring a complete redesign of the system. Future enhancements may improve usability, experiment management, visualization, and learning performance while preserving the overall structure of the implementation.

The proposed enhancements presented in this chapter are grouped into three categories:

* **Software and Usability Enhancements**, focusing on improving the user experience and simplifying experimentation.
* **Experimentation and Analysis Enhancements**, aimed at making training runs easier to monitor, compare, and reproduce.
* **Advanced Reinforcement Learning Enhancements**, introducing well-established improvements to the original DQN algorithm.

These enhancements represent a natural evolution of the current project, progressing from improvements in software engineering practices to more sophisticated reinforcement learning techniques.

---

# 14.1 Software and Usability Enhancements

Several practical improvements could make the project easier to use and more convenient for conducting reinforcement learning experiments.

### Interactive Experiment Dashboard

Currently, training progress is monitored through console output and static plots generated after training has completed.

A future version could provide a lightweight graphical dashboard displaying information such as:

* current episode,
* episode reward,
* current loss,
* epsilon value,
* replay memory utilization,
* target network synchronization events,
* recent training plots,
* latest evaluation video,
* current configuration values.

Such a dashboard would allow experiments to be monitored in real time and would greatly improve the usability of the project.

---

### Resume Training

Version 1.0 always begins training from randomly initialized network weights.

A useful enhancement would allow training to resume from an existing checkpoint.

This capability would make it possible to:

* continue interrupted training sessions,
* extend previously trained models,
* perform long-running experiments over multiple sessions.

---

### Automatic Checkpoint Management

The current implementation maintains a single model checkpoint.

Future versions could automatically manage multiple checkpoints, for example:

* latest model,
* best-performing model,
* periodic checkpoints (e.g., every 100 episodes).

This would simplify model comparison and reduce the risk of overwriting valuable checkpoints.

---

### Richer Gameplay Videos

Evaluation videos could be enhanced by displaying additional runtime information directly on the recorded gameplay.

Possible overlays include:

* current episode,
* predicted action,
* accumulated reward,
* predicted Q-value,
* elapsed time,
* evaluation configuration.

These additions would improve both debugging and presentation quality.

---

# 14.2 Experimentation and Analysis Enhancements

As reinforcement learning experiments become larger, organizing and comparing results becomes increasingly important.

Several improvements could enhance the project's experiment management capabilities.

### Experiment Management

Rather than storing all outputs in a common directory, future versions could automatically create dedicated experiment folders.

Each experiment could contain:

* configuration file,
* model checkpoints,
* training log,
* learning curves,
* evaluation videos,
* summary reports.

This organization would simplify comparison between different training runs.

---

### Enhanced Evaluation Metrics

Version 1.0 evaluates a single gameplay episode.

Future versions could automatically execute multiple evaluation episodes and report summary statistics including:

* mean reward,
* best reward,
* worst reward,
* standard deviation,
* average episode length,
* average distance traveled.

These metrics would provide a more reliable estimate of the agent's true performance.

---

### Improved Training Visualizations

The current plotting utilities generate reward, loss, and epsilon curves.

Additional visualizations could include:

* moving average reward,
* cumulative reward,
* replay memory growth,
* target network update markers,
* smoothed loss curves.

These plots would make training behavior easier to interpret and facilitate comparison between experiments.

---

### Automated Experiment Reports

Following each training session, the project could automatically generate a summary report containing:

* configuration values,
* final performance metrics,
* training plots,
* evaluation results,
* links to generated videos.

Such reports would provide a convenient record of completed experiments and improve reproducibility.


## 14.3 Advanced Reinforcement Learning Enhancements

While Version 1.0 implements the original Deep Q-Network (DQN) algorithm, several well-established extensions could improve learning performance while preserving the overall architecture of the project.

These enhancements represent natural next steps for future versions.

---

### Double Deep Q-Network (Double DQN)

One limitation of the original DQN algorithm is its tendency to **overestimate action values**. Since the same network is used both to select and evaluate actions, the predicted Q-values can become overly optimistic, leading to unstable learning.

Double DQN addresses this issue by separating action selection from action evaluation during the Bellman update.

Potential benefits include:

* Reduced overestimation of Q-values
* Improved training stability
* More reliable value estimation
* Better overall gameplay performance

Because the current implementation already maintains separate online and target networks, incorporating Double DQN would require only moderate modifications to the learning algorithm.

---

### Prioritized Experience Replay

Version 1.0 samples experiences uniformly from replay memory.

In practice, however, not all experiences are equally informative. Experiences with larger prediction errors often contribute more to learning than those the agent already understands well.

Prioritized Experience Replay assigns a higher probability of sampling these more informative experiences.

Potential benefits include:

* Faster convergence
* Improved sample efficiency
* Better use of replay memory
* Reduced training time

Implementing this enhancement would primarily require replacing the current uniform sampling strategy with a priority-based sampling mechanism.

---

### Dueling Deep Q-Network (Dueling DQN)

The current neural network directly predicts one Q-value for each possible action.

A Dueling DQN separates this prediction into two components:

* **State Value** — how desirable the current state is.
* **Action Advantage** — how much better one action is compared to the others in that state.

These two quantities are then combined to produce the final Q-values.

This architecture often performs better in environments where many actions produce similar outcomes.

Potential advantages include:

* Improved learning efficiency
* Better state representation
* Faster convergence
* Enhanced decision making in complex situations

Because the overall training pipeline remains unchanged, only the neural network architecture would require modification.

---

### Summary

These enhancements build directly upon the existing Version 1.0 implementation without requiring a complete redesign of the project.

Collectively, they represent a logical roadmap for Version 2.0, allowing the project to evolve from a classical DQN implementation into a more capable and robust reinforcement learning agent while preserving its modular architecture and overall design philosophy.


---
# 15. References

The implementation presented in this project is based on established reinforcement learning concepts and publicly available software libraries. The following references provide useful background for readers wishing to explore the underlying algorithms and technologies in greater depth.

---

# Books

## Sutton, R. S., & Barto, A. G. (2018)

**Reinforcement Learning: An Introduction (2nd Edition)**

The definitive introductory text on reinforcement learning, covering Markov Decision Processes, temporal-difference learning, Q-learning, policy gradients, and many other fundamental topics.

> Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

Available online:

http://incompleteideas.net/book/the-book-2nd.html

---

## Goodfellow, I., Bengio, Y., & Courville, A. (2016)

**Deep Learning**

A comprehensive reference covering neural networks, optimization, convolutional networks, and deep learning techniques used throughout this project.

> Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

Available online:

https://www.deeplearningbook.org/

---

# Research Papers

## Human-level Control through Deep Reinforcement Learning

This landmark paper introduced the Deep Q-Network (DQN) algorithm developed by DeepMind.

> Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). *Human-level Control through Deep Reinforcement Learning*. Nature, 518(7540), 529–533.

---

## Playing Atari with Deep Reinforcement Learning

The original DQN technical report introducing experience replay and target networks.

> Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2013). *Playing Atari with Deep Reinforcement Learning*. arXiv:1312.5602.

---

# Software Libraries

## PyTorch

Deep learning framework used to implement the neural network.

https://pytorch.org/

---

## Gym

Standard reinforcement learning interface used by the project.

Although Gym is now deprecated, it remains the API used by Gym Super Mario Bros.

https://www.gymlibrary.dev/

---

## Gymnasium

The actively maintained successor to Gym.

Future versions of this project may migrate to Gymnasium.

https://gymnasium.farama.org/

---

## Gym Super Mario Bros

Provides the Super Mario Bros reinforcement learning environment.

https://github.com/Kautenja/gym-super-mario-bros

---

## NES-Py

NES emulator used internally by Gym Super Mario Bros.

https://github.com/Kautenja/nes-py

---

## OpenCV

Used for image preprocessing, including grayscale conversion and image resizing.

https://opencv.org/

---

## NumPy

Provides numerical array operations used throughout the project.

https://numpy.org/

---

## Pandas

Used to read and analyze training logs.

https://pandas.pydata.org/

---

## Matplotlib

Used to generate learning curves and other visualizations.

https://matplotlib.org/

---

# Development Tools

The project was developed using:

* Visual Studio Code
* Conda
* Python 3.10
* NVIDIA CUDA
* Git (recommended for version control)

---

# Additional Learning Resources

The following topics are recommended for readers interested in extending this project.

## Reinforcement Learning

* Double Deep Q-Network (Double DQN)
* Dueling Networks
* Prioritized Experience Replay
* Rainbow DQN
* Actor-Critic Methods
* Proximal Policy Optimization (PPO)
* Soft Actor-Critic (SAC)

---

## PyTorch

Recommended areas for further study include:

* custom datasets,
* GPU optimization,
* tensor operations,
* model checkpointing,
* learning rate schedulers.

---

## Computer Vision

Relevant topics include:

* convolutional neural networks,
* image augmentation,
* feature extraction,
* transfer learning.

---

# Suggested Next Steps

Readers wishing to extend this project may consider implementing one or more of the following enhancements:

1. Double DQN.
2. Dueling DQN.
3. Prioritized Experience Replay.
4. TensorBoard integration.
5. Automatic checkpoint selection.
6. Evaluation across multiple episodes.
7. Hyperparameter optimization.
8. Migration from Gym to Gymnasium.

These extensions build naturally upon the modular architecture developed in Version 1.0.

---

# Repository Notes

The source code is organized to emphasize:

* readability,
* modularity,
* reproducibility,
* incremental development,
* experimentation.

The accompanying README is intended to serve not only as project documentation but also as a learning guide explaining the rationale behind each major design decision.

---

