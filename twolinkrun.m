clear
clc
close all

mdl = 'twolinkenv';
open_system(mdl)

agentBlk = [mdl '/RL Agent'];

obsInfo = rlNumericSpec([6 1],...
    'LowerLimit',[-1 -1 -inf -1 -1 -inf]',...
    'UpperLimit',[1 1 inf 1 1 inf]');
obsInfo.Name = 'observations';


actInfo = rlNumericSpec([2 1],...
    'LowerLimit',[-15;-15],...
    'UpperLimit',[15;15]);
actInfo.Name = 'torque';


env = rlSimulinkEnv(mdl,agentBlk,obsInfo,actInfo);

Ts = 0.02;
Tf = 25;

rng(0)


statePath = [
    imageInputLayer([6 1 1], 'Normalization', 'none', 'Name', 'observation')
    fullyConnectedLayer(128, 'Name', 'CriticStateFC1')
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(200, 'Name', 'CriticStateFC2')];

actionPath = [
    imageInputLayer([2 1 1], 'Normalization', 'none', 'Name', 'action')
    fullyConnectedLayer(200, 'Name', 'CriticActionFC1', 'BiasLearnRateFactor', 0)];

commonPath = [
    additionLayer(2,'Name', 'add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1, 'Name', 'CriticOutput')];

criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);
    
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

criticOptions = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1,'UseDevice',"gpu");

critic = rlRepresentation(criticNetwork,criticOptions,'Observation',{'observation'},obsInfo,'Action',{'action'},actInfo);

actorNetwork = [
    imageInputLayer([6 1 1], 'Normalization', 'none', 'Name', 'observation')
    fullyConnectedLayer(128, 'Name', 'ActorFC1')
    reluLayer('Name', 'ActorRelu1')
    fullyConnectedLayer(200, 'Name', 'ActorFC2')
    reluLayer('Name', 'ActorRelu2')
    fullyConnectedLayer(2, 'Name', 'ActorFC3')
    tanhLayer('Name', 'ActorTanh1')
    scalingLayer('Name','ActorScaling','Scale',reshape([15;15],[1,1,2]))];

actorOptions = rlRepresentationOptions('LearnRate',5e-04,'GradientThreshold',1,'UseDevice',"gpu");

actor = rlRepresentation(actorNetwork,actorOptions,'Observation',{'observation'},obsInfo,'Action',{'ActorScaling'},actInfo);

agentOptions = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e6,...
    'MiniBatchSize',128);
agentOptions.NoiseOptions.Variance = [0.4;0.4];
agentOptions.NoiseOptions.VarianceDecayRate = 1e-5;

agent = rlDDPGAgent(actor,critic,agentOptions);

maxepisodes = 2000;
maxsteps = ceil(Tf/Ts);
trainingOptions = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes,...
    'MaxStepsPerEpisode',maxsteps,...
    'ScoreAveragingWindowLength',5,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',0,...
    'SaveAgentCriteria','EpisodeReward',...
    'SaveAgentValue',0);

trainingStats = train(agent,env,trainingOptions);


