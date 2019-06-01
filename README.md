# Two-Link Robotic Arm Control using DDPG

This project implements a Deep Deterministic Policy Gradient (DDPG) reinforcement learning agent to control a two-link robotic arm in a Simulink environment. The agent learns to apply appropriate torques to the arm's joints to achieve desired control objectives.

## Project Structure

- `twolinkrun.m`: Main MATLAB script that sets up and trains the DDPG agent
- `twolinkenv.slx`: Simulink model of the two-link robotic arm environment

## Requirements

- MATLAB R2020b or later
- Deep Learning Toolbox
- Reinforcement Learning Toolbox
- Simulink
- GPU (recommended for faster training)

## Setup

1. Clone or download this repository
2. Open MATLAB and navigate to the project directory
3. Open `twolinkrun.m` in the MATLAB editor

## Training the Agent

To train the DDPG agent, simply run the `twolinkrun.m` script in MATLAB. The script will:

1. Initialize the two-link arm environment
2. Create actor and critic neural networks
3. Set up the DDPG agent with appropriate hyperparameters
4. Train the agent for up to 2000 episodes

The training progress will be displayed in the Reinforcement Learning Episode Manager, showing the episode reward over time.

## Agent Configuration

### Observation Space
- 6-dimensional vector containing joint angles and velocities
- Normalized between [-1, 1] for angular positions

### Action Space
- 2-dimensional vector representing torques applied to each joint
- Torque range: ±15 Nm for each joint

### Network Architecture

**Actor Network:**
- Input: 6D state vector
- Hidden layers: 128 and 200 neurons with ReLU activation
- Output: 2D action vector with tanh activation, scaled to ±15 Nm

**Critic Network:**
- State path: 6D input → 128 → 200 neurons
- Action path: 2D input → 200 neurons
- Combined path: Addition of state and action paths → ReLU → 1D Q-value

## Training Parameters

- Sample time: 0.02 seconds
- Episode length: 25 seconds
- Experience buffer: 1,000,000 steps
- Mini-batch size: 128
- Actor learning rate: 5e-4
- Critic learning rate: 1e-3
- Target update factor: 1e-3
- Exploration noise: Initial variance of 0.4 with decay rate of 1e-5

## Visualization

During training, the following metrics are tracked:
- Episode reward
- Average reward
- Q0 value estimate

## Notes

- The environment is implemented in Simulink (`twolinkenv.slx`)
- Training uses GPU acceleration if available
- The best agent is saved based on episode reward


