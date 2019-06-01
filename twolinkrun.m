%% Clear Workspace and Close Figures
clear        % Clear workspace variables
clc          % Clear command window
close all    % Close all open figures

%% Load and Open Simulink Model
mdl = 'twolinkenv';      % Name of the Simulink model
open_system(mdl)         % Open the Simulink model in the background

% Define the path to the RL Agent block in the Simulink model
agentBlk = [mdl '/RL Agent'];

%% Define Observation Space
% Create observation specification for the RL environment
% The state vector contains 6 elements: [q1, q2, q1_dot, q2_dot, x, y]
% where q1, q2 are joint angles and x, y are end-effector positions
obsInfo = rlNumericSpec([6 1],...
    'LowerLimit',[-1 -1 -inf -1 -1 -inf]',...  % Limits on joint angles and velocities
    'UpperLimit',[1 1 inf 1 1 inf]');          % Normalized angles between [-1, 1]
obsInfo.Name = 'observations';                  % Name for the observation space

%% Define Action Space
% Create action specification for the RL environment
% The action vector contains 2 elements: [tau1, tau2]
% where tau1 and tau2 are torques applied to each joint
actInfo = rlNumericSpec([2 1],...
    'LowerLimit',[-15;-15],...  % Minimum torque for each joint (-15 Nm)
    'UpperLimit',[15;15]);      % Maximum torque for each joint (15 Nm)
actInfo.Name = 'torque';        % Name for the action space

% Create RL environment interface with the Simulink model
env = rlSimulinkEnv(mdl, agentBlk, obsInfo, actInfo);

%% Define Simulation Parameters
Ts = 0.02;  % Sample time (seconds)
Tf = 25;    % Simulation time (seconds)

rng(0)  % Set random seed for reproducibility

%% Create Critic Network
% State path processes the observation (state) input
statePath = [
    imageInputLayer([6 1 1], 'Normalization', 'none', 'Name', 'observation')  % Input layer for state
    fullyConnectedLayer(128, 'Name', 'CriticStateFC1')  % First hidden layer
    reluLayer('Name', 'CriticRelu1')                   % ReLU activation
    fullyConnectedLayer(200, 'Name', 'CriticStateFC2')]; % Second hidden layer

% Action path processes the action input
actionPath = [
    imageInputLayer([2 1 1], 'Normalization', 'none', 'Name', 'action')  % Input layer for action
    fullyConnectedLayer(200, 'Name', 'CriticActionFC1', 'BiasLearnRateFactor', 0)];  % Hidden layer

% Common path combines state and action paths
commonPath = [
    additionLayer(2,'Name', 'add')                % Add state and action paths
    reluLayer('Name','CriticCommonRelu')          % ReLU activation
    fullyConnectedLayer(1, 'Name', 'CriticOutput')];  % Output Q-value

% Assemble the critic network
criticNetwork = layerGraph(statePath);  % Initialize with state path
criticNetwork = addLayers(criticNetwork, actionPath);  % Add action path
criticNetwork = addLayers(criticNetwork, commonPath);  % Add common path
    
% Connect the state and action paths to the addition layer
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

% Configure critic options
criticOptions = rlRepresentationOptions('LearnRate',1e-03,...  % Learning rate
                                      'GradientThreshold',1,...  % Gradient clipping threshold
                                      'UseDevice',"gpu");       % Use GPU if available

% Create critic representation
critic = rlRepresentation(criticNetwork, criticOptions,...
                         'Observation',{'observation'}, obsInfo,...
                         'Action', {'action'}, actInfo);

%% Create Actor Network
% The actor network maps states to actions (policy function)
actorNetwork = [
    % Input layer for state observations
    imageInputLayer([6 1 1], 'Normalization', 'none', 'Name', 'observation')
    
    % Hidden layers with ReLU activation
    fullyConnectedLayer(128, 'Name', 'ActorFC1')
    reluLayer('Name', 'ActorRelu1')
    fullyConnectedLayer(200, 'Name', 'ActorFC2')
    reluLayer('Name', 'ActorRelu2')
    
    % Output layer with tanh activation (bounded between -1 and 1)
    fullyConnectedLayer(2, 'Name', 'ActorFC3')
    tanhLayer('Name', 'ActorTanh1')
    
    % Scale the output to the action space range (-15 to 15 Nm)
    scalingLayer('Name','ActorScaling','Scale',reshape([15;15],[1,1,2]))];

% Configure actor options
actorOptions = rlRepresentationOptions('LearnRate',5e-04,...  % Learning rate
                                     'GradientThreshold',1,...  % Gradient clipping threshold
                                     'UseDevice',"gpu");       % Use GPU if available

% Create actor representation
actor = rlRepresentation(actorNetwork, actorOptions,...
                        'Observation',{'observation'}, obsInfo,...
                        'Action', {'ActorScaling'}, actInfo);

%% Configure DDPG Agent Options
agentOptions = rlDDPGAgentOptions(...
    'SampleTime', Ts, ...              % Time between control actions
    'TargetSmoothFactor', 1e-3, ...     % Soft target update factor
    'ExperienceBufferLength', 1e6, ...  % Size of experience buffer
    'MiniBatchSize', 128);              % Mini-batch size for training

% Configure exploration noise
agentOptions.NoiseOptions.Variance = [0.4; 0.4];           % Initial exploration noise
agentOptions.NoiseOptions.VarianceDecayRate = 1e-5;         % Noise decay rate

% Create DDPG agent with actor, critic, and options
agent = rlDDPGAgent(actor, critic, agentOptions);

%% Configure Training Options
maxepisodes = 2000;              % Maximum number of training episodes
maxsteps = ceil(Tf/Ts);          % Maximum steps per episode

trainingOptions = rlTrainingOptions(...
    'MaxEpisodes', maxepisodes, ...                % Stop after max episodes
    'MaxStepsPerEpisode', maxsteps, ...            % Maximum steps per episode
    'ScoreAveragingWindowLength', 5, ...           % Window size for averaging episode rewards
    'Verbose', false, ...                          % Display training progress in command window
    'Plots', 'training-progress', ...              % Show training progress plot
    'StopTrainingCriteria', 'AverageReward', ...   % Stop when average reward reaches threshold
    'StopTrainingValue', 0, ...                    % Threshold for stopping training
    'SaveAgentCriteria', 'EpisodeReward', ...      % Save agent based on episode reward
    'SaveAgentValue', 0);                         % Save agent when reward exceeds this value

%% Train the Agent
% This will train the agent using the specified options and environment
trainingStats = train(agent, env, trainingOptions);

% After training, the agent can be used for inference or saved for later use
% save('trainedAgent.mat', 'agent');  % Uncomment to save the trained agent


