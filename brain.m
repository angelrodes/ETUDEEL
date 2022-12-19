function [ outputs,cost ] = brain(neural_network_options,parameters,inputs,varargin)
% BRAIN run a neural network with the given parameters and inputs
%   and output the corresponing output values and, if objectives are given,
%   cost is calculated as the mean squeared deviation
%  Angel Rodes, 2022

% parameters(neural_network_options.bias_index(neural_network_options.bias_index>0))=0; % no bias

%% Get objectives (if any) or consider zeros
if ~isempty(varargin)
    objectives=varargin{1};
else
    objectives=zeros(1,neural_network_options.n_outputs,'single');
end

% check parameters size
if neural_network_options.n_parameters>size(parameters,2)
    error('Not enough parameters')
elseif neural_network_options.n_parameters<numel(parameters)
    warning('Too many paremeters')
end

% Calculate neuron values 
values=zeros(size(inputs,1),neural_network_options.n_neurons);
values(1:size(inputs,1),1:neural_network_options.n_input_neurons)=inputs;
for neuron=neural_network_options.n_input_neurons+1:neural_network_options.n_neurons
    parent_neurons=...
        neural_network_options.layer_index==(neural_network_options.layer_index(neuron)-1);
    weight_indexes=...
        neural_network_options.weight_index_min(neuron):neural_network_options.weight_index_max(neuron);
    values(1:size(inputs,1),neuron)=neural_network_options.hidden_activation_function(...
        sum(values(1:size(inputs,1),parent_neurons).*parameters(weight_indexes),2)...
        +parameters(neural_network_options.bias_index(neuron))...
        );
end



outputs=values(1:size(inputs,1),...
    neural_network_options.n_neurons-neural_network_options.n_output_neurons+1:neural_network_options.n_neurons);
cost=sum(sum((outputs-objectives).^2))/numel(outputs);

end

