%% Training the neural network (brain)
%%  Angel Rodes, 2022

% TO DO
% add diaglogs

%% init
clear
close all hidden
diary('output.txt')

%% Inputs
% input dialog !!!!
% data_file='lat_lon_elv_concentration_age.csv';
data_file='primary_Be_lat_lon_alt_thick_den_shield_year_Be-10_dBe-10_TrueAge.csv'; % fracaso
% data_file='x_log10x.csv';

%% Check file
% secondary_data_file='secondary_Be_lat_lon_alt_thick_den_shield_year_Be-10_dBe-10_TrueAge.csv';
secondary_data_file=data_file;

% read input
learning_data=csvread(data_file);
secondary_learning_data=csvread(secondary_data_file);

% intially consider only one output
n_input_columns=size(learning_data,2)-1;
n_objective_columns=1;

%% Naural and fitting options
n_generations=100e3; % number of generations
samples_per_generation=min(5e3,max(ceil(max(size(learning_data,1)/10)),min(100,size(learning_data,1)))); %number of input data used for each generation

n_hidden_layers=2; % number of layers in the neural network
n_extra_neurons_per_layer=ceil(size(learning_data,2)/5); % number of neurons extra neurons in first hidden layer

epsilon_max=1; % maximum ε/parameter
change_only_one_parameter=0.5; % if 1, only one parameter is changed in each generation
percent_parameters_to_change=1/n_hidden_layers; % percentage of parameters to fit each generation when changing more than one paratmeter
n_initial_learning_generations=round(n_generations*0.1); % genereations with lots of mutants and no pruning

use_bias=1; % 0 or 1: (not) to use bias parameters
prune_neurons=1; % pruning or not

allow_log_conversion=1; % allow considering input/output data in the log scale

layer_structure=...
    [n_input_columns,...
    max([n_input_columns,n_objective_columns]+n_extra_neurons_per_layer)*ones(1,n_hidden_layers>0),...
    round(max([n_input_columns,n_objective_columns]))*ones(1,n_hidden_layers-1),...
    n_objective_columns]; 

% options dialog !!!!
% recalculate parameters

%% Check, and organise learning data
% learning_data=csvread(data_file);

% Check data and columns
extra_columns=size(learning_data,2)-n_input_columns-n_objective_columns;
if extra_columns>0
    warning(['Right ' num2str(extra_columns) ' columns in ' data_file ' will not be used!'])
elseif extra_columns<0
    error(['Only ' num2str(size(learning_data,2)) ' columns in ' data_file...
        '. Neural structure requires ' num2str(layer_structure(1)) ' input and '...
        num2str(layer_structure(1)) ' output columns!'])
end

% get inputs and outputs
raw_inputs=learning_data(:,1:n_input_columns);
raw_objectives=learning_data(:,n_input_columns+1:n_input_columns+n_objective_columns);

%% Transform learning data to  input/output values in the 0-1 range
% inputs=zeros(size(raw_inputs,1),size(raw_inputs,2)*2)+NaN;
% objectives=zeros(size(raw_objectives,1),size(raw_objectives,2)*2)+NaN;
inputs=zeros(size(raw_inputs,1),size(raw_inputs,2)*1)+NaN;
objectives=zeros(size(raw_objectives,1),size(raw_objectives,2)*1)+NaN;

% Model selection

for n=1:n_input_columns+n_objective_columns
    if n<=n_input_columns
        data=raw_inputs(:,n);
    else
        data=raw_objectives(:,n-n_input_columns);
    end
    normal_p=abs(mean(data)-median(data))/max(mean(data),median(data));
    log_p=abs(mean(log(data))-median(log(data)))/max(mean(log(data)),median(log(data)));
    if log_p<normal_p && min(data)>0 && allow_log_conversion>0
        log_distribution(n)=1;
        min_data(n)=min(log(data));
        range_data(n)=range(log(data));
    else
        log_distribution(n)=0;
        min_data(n)=min(data);
        range_data(n)=range(data);
    end
end



% Input and output functions
% transform_function=@(x)(...
%     atan((2*x)./(x.^2-1))...
%     )/(2*pi)+...
%     0.5.*(x<1)+0.5.*(x<=-1); % stereographic
% detransform_function=@(t)...
%     (1+(1+tan(t*2*pi).^2).^0.5)./tan(t*2*pi).*(t<=0.25)+...
%     (1-(1+tan(t*2*pi).^2).^0.5)./tan(t*2*pi).*(t>0.25 & t<=0.75)+...
%     (1+(1+tan(t*2*pi).^2).^0.5)./tan(t*2*pi).*(t>0.75); % stereographic
transform_function=@(column,data)...
    (log_distribution(column)==1)*(log(data)-min_data(column))./range_data(column)+...
    (log_distribution(column)==0)*(   (data)-min_data(column))./range_data(column); % linear 0-1
detransform_function=@(column,data)...
    (log_distribution(column)==1)*exp((log_distribution(column)==1)*(data*range_data(column)+min_data(column)))+...
    (log_distribution(column)==0)*   (data*range_data(column)+min_data(column)); % linear 0-1

% Transform to 0-1
figure('units','normalized','outerposition',[0 0 1 1])
for n=1:n_input_columns+n_objective_columns
     subplot(2,max(n_input_columns,n_objective_columns),n)
    if n<=n_input_columns
        inputs(:,n)=transform_function(n,raw_inputs(:,n));
        hist(inputs(:,n),linspace(0,1,20))
        xlabel(['NN input ' num2str(n)])
    else
        objectives(:,n-n_input_columns)=transform_function(n,raw_objectives(:,n-n_input_columns));
        hist(objectives(:,n-n_input_columns),linspace(0,1,20))
        xlabel(['NN output ' num2str(n-n_input_columns)])
    end
    
    xlim([0 1])
end
drawnow;
% pause(5) % have a look to the distribution of tranformed inputs/outputs for 5 sec
% close all hidden

%% Build neural network
% n_hidden_layers=5;
n_hidden_layers=single(n_hidden_layers);

n_tranformed_inputs=size(inputs,2);
n_transformed_objectives=size(objectives,2);

% layer_structure=...
%     [n_tranformed_inputs,...
%     max([n_tranformed_inputs,n_transformed_objectives]+n_extra_neurons_per_layer)*ones(1,n_hidden_layers),...
%     n_transformed_objectives];

n_weights=sum(layer_structure(2:end).*layer_structure(1:end-1));
n_bias=sum(layer_structure(2:end));
n_parameters=n_weights+n_bias;

% Indexes
n_neurons=sum(layer_structure);
layer_index=zeros(1,n_neurons);
weight_index_min=zeros(1,n_neurons);
weight_index_max=zeros(1,n_neurons);
bias_index=zeros(1,n_neurons);
neuron_position=0;
weight_position=0;
bias_position=n_weights;
for layer=1:numel(layer_structure)
    for neuron=1:layer_structure(layer)
        neuron_position=neuron_position+1;
        layer_index(neuron_position)=layer;
        if layer>1
            bias_position=bias_position+1;
            bias_index(neuron_position)=bias_position;
            weight_position=weight_position+1;
            weight_index_min(neuron_position)=weight_position;
            weight_position=weight_position-1+layer_structure(layer-1);
            weight_index_max(neuron_position)=weight_position;
        end
    end
end
neural_network_options.layer_index=layer_index;
neural_network_options.bias_index=bias_index;
neural_network_options.weight_index_min=weight_index_min;
neural_network_options.weight_index_max=weight_index_max;

neural_network_options.layer_structure=layer_structure;
neural_network_options.n_neurons=sum(layer_structure);
neural_network_options.n_input_neurons=layer_structure(1);
neural_network_options.n_output_neurons=layer_structure(end);
neural_network_options.n_weights=sum(layer_structure(2:end).*layer_structure(1:end-1));
neural_network_options.n_bias=sum(layer_structure(2:end));
neural_network_options.n_parameters=n_weights+n_bias;

% activation function(s)
neural_network_options.hidden_activation_function=@(z)...
    1./(1+exp(-z)); % sigmoid

if neural_network_options.n_parameters>size(learning_data,1)
    warning('More parameters than data. Clearly overfitting!')
end

%% Train

% initial parameter array
par_index=[ones(1,n_weights+n_bias*use_bias),zeros(1,n_bias*(use_bias==0))]; % 1 for weights and 1 or 0 for bias

% build indexes for parent neuron of each weight (for pruning)
parent_neuron_index=par_index.*0;
idx=0;
previous_neurons=0;
for layer=1:max(layer_index)-1
    for next_layer_count=1:sum(layer_index==layer+1)
        for n=1:sum(layer_index==layer)
            idx=idx+1;
            parent_neuron_index(idx)=n+previous_neurons;
        end
    end
    previous_neurons=previous_neurons+sum(layer_index==layer);
end
previous_neurons=sum(layer_index==1);
for layer=2:max(layer_index)
    for n=1:sum(layer_index==layer)
        idx=idx+1;
        if layer==max(layer_index)
            parent_neuron_index(idx)=0; % do not prune the output layer
        else
            %  parent_neuron_index(idx)=n+previous_neurons;
            parent_neuron_index(idx)=0; % ignore bias
        end
    end
    previous_neurons=previous_neurons+sum(layer_index==layer);
end

% seeder function
seed_generator=@(par_idx)...
    rand(size(par_idx)).*par_idx; % between -1 and 1 
% check time
tic

% start plotting
figure('units','normalized','outerposition',[0.5 0 0.5 1])
subplot(2,1,1)
hold on
xlabel('generation')
ylabel('MSE')
% title(['Brain structure: ' char(8594) ' ' num2str(layer_structure) ' ' char(8594)])
set(gca, 'YScale', 'log')
% set(gca, 'XScale', 'log')
grid on
box on
text(0,0.5^2,[' ' char(8592) ' monkey guess'])
draw_each=100; % draw restults every draw_each generations (starting value)

% init variables
cost_values=zeros(1,n_generations).*NaN;
fitting_type=zeros(1,n_generations);

% seed
parameters=seed_generator(par_index);

% iterate
for generation=1:n_generations
    % random sample data
    if size(inputs,1)>=samples_per_generation
        random_line_order=randperm(size(inputs,1));
        random_lines=random_line_order(1:samples_per_generation);
    else
        random_lines=floor(rand(samples_per_generation,1)*size(inputs,1)+1);
    end
    inputs_i=inputs(random_lines,:);
    objectives_i=objectives(random_lines,:);
    
    % check model
    [output_values,cost_i]=brain(neural_network_options,parameters,inputs_i,objectives_i);
    cost_values(generation)=cost_i;
    
    % parameters +/- epsilon
    if rand<change_only_one_parameter
        random_order=randperm(numel(par_index));
        candidates=random_order.*par_index(random_order);
        candidates=candidates(candidates>0);
        parameter_to_change=candidates(1);
%         parameter_to_change=floor(rand*(n_weights+n_bias*use_bias)-1e-10)+1; % only one
        parameters_to_change=( 1:numel(parameters)==parameter_to_change );
    else
        parameters_to_change=rand(size(parameters))>percent_parameters_to_change;
    end
    epsilon=epsilon_max*cost_i^0.5.*seed_generator(par_index).*parameters_to_change.*parameters;
%         epsilon=epsilon_max.*rand.*seed_generator().*parameters_to_change;
    [~,cost_i_plus]=brain(neural_network_options,parameters+epsilon,inputs_i,objectives_i);
    [~,cost_i_minus]=brain(neural_network_options,parameters-epsilon,inputs_i,objectives_i);
    
    % guess minimum using Newton-ish method
    if numel(unique([cost_i_minus,cost_i,cost_i_plus]))>1
        newton_position=newton_ish_method(cost_i_minus,cost_i,cost_i_plus);
    else
        newton_position=0;
    end
    newton_parameters=parameters+epsilon.*newton_position;
    [~,cost_i_newton]=brain(neural_network_options,newton_parameters,inputs_i,objectives_i);
    
    % mutant
    if generation>n_initial_learning_generations
        mutant_parameters=seed_generator(par_index).*parameters_to_change+...
            parameters.*~parameters_to_change;
    else % at the beguinning mutate all
        mutant_parameters=seed_generator(par_index);
    end
    [~,cost_i_mutant]=brain(neural_network_options,mutant_parameters,inputs_i,objectives_i);
    
    % select best parameters
    best_cost=min([cost_i,cost_i_plus,cost_i_minus,cost_i_newton,cost_i_mutant]);
    if  cost_i_newton==best_cost
        parameters=newton_parameters;
        fitting_type(generation)=1;
        %         disp(['Succesful Newton in generation ' num2str(generation)])
    elseif cost_i_mutant==best_cost
        parameters=mutant_parameters;
        fitting_type(generation)=2;
        %         disp(['Succesful mutant in generation ' num2str(generation)])
    elseif cost_i_plus==best_cost
        parameters=parameters+epsilon;
        fitting_type(generation)=3;
        %         disp(['Succesful epsilon in generation ' num2str(generation)])
    elseif cost_i_minus==best_cost
        parameters=parameters-epsilon;
        fitting_type(generation)=3;
        %         disp(['Succesful epsilon in generation ' num2str(generation)])
    else
        %         disp(['Nothing changed in generation ' num2str(generation)])
    end
    
    parameters=parameters.*par_index; % prune
    
    % plot and prune every now and then
    if mod(generation,draw_each)==0 || generation==n_generations
        
        time_elapsed=toc;
        time_total=time_elapsed/generation*n_generations;
        eta_minutes=round((time_total-time_elapsed)/60);
        draw_each=ceil(5*generation/time_elapsed); % draw each 5 seconds
        
        number_of_unique_outputs=numel(unique(output_values))/size(output_values,2);
        
        % update plot
        subplot(2,1,1)
        hold on
        plot_indexes=max(1,generation-draw_each+1):generation;
        plot(plot_indexes,cost_values(plot_indexes),'.b')
        ylim([10^floor(log10(min(cost_values(1:generation)))) max(1,cost_i)])
        xlim([0 n_generations])
        
        % plot brain
        subplot(2,1,2)
        plot_brain( neural_network_options,parameters )
        hold on
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        % axis equal
        xlabel(['ETA = ' num2str(eta_minutes) ' minutes'])
        
        drawnow;
        
        % prune
        if generation>n_initial_learning_generations && prune_neurons==1
            for n=1:max(parent_neuron_index)
                if numel(parameters(parent_neuron_index==n))>0
                    if max(abs(parameters(parent_neuron_index==n)))<cost_i % REFINE THIS CRITERIA?
                        par_index(parent_neuron_index==n)=0;
                        parent_neuron_index(parent_neuron_index==n)=0;
                        par_index(weight_index_min(n):weight_index_max(n))=0;
                        par_index(bias_index(neuron))=0;
                        disp(['Neuron ' num2str(n) ' prunned in generation ' num2str(generation)])
                        n_initial_learning_generations=n_initial_learning_generations+generation; 
                    end
                end
            end
        end
    
    end
end
[~,cost_i]=brain(neural_network_options,parameters,inputs_i,objectives_i);
subplot(2,1,1)
hold on
text(generation,cost_i,[' ' num2str(cost_i) ' (last)'],'Color','k')

%% Calulate model outputs
[model_values,cost_global]=brain(neural_network_options,parameters,inputs,objectives);

text(generation/2,cost_global,['Full dataset = ' num2str(cost_global)],...
    'Color','r','VerticalAlignment','Bottom','BackgroundColor','w')

raw_model_values=raw_objectives.*NaN;
for n=1:n_objective_columns
    raw_model_values(:,n)=detransform_function(n+n_input_columns,model_values(:,n));
end


%% Plot approximations
for column=1:size(raw_objectives,2)
    figure('units','normalized','outerposition',[0 0 0.5 0.5])
    hold on
%     title(['Objetive ' num2str(column)])
    xlabel('Data')
    ylabel('AI')
    data=raw_objectives(:,column);
    model=raw_model_values(:,column);
    
    valid=(abs(data)>0)&(abs(model)>0);
    average_uncert=mean(abs(data(valid)-model(valid))./data(valid));
    title(['Objetive' num2str(column) ' Av. uncert.=' num2str(average_uncert*100,3) '%'])
    
    plot(data,model,'.b')
    plot([min(data),max(data)],[min(data),max(data)],'--g')
    
    if log_distribution(n_input_columns+column)==1
        set(gca, 'YScale', 'log')
        set(gca, 'XScale', 'log')
    end
    box on 
    grid on
end


%% Dispalay results
disp('------')
disp('Results:')
disp(' ')

disp(['File: ' data_file])
disp(['Brain structure: ' char(8594) ' ' num2str(layer_structure) ' ' char(8594)])
disp(['MSE = ' num2str(cost_global,3)]);
disp(' ')

% disp('Loss function:')
% disp([ 'Mean: ' num2str(mean(cost_values(1:generation)))]);
% disp([ 'Median: ' num2str(median(cost_values(1:generation)))]);
% disp([ 'Last: ' num2str(cost_i)]);
% disp([ 'Average global: ' num2str(cost_global)]);
% disp(' ')

disp('Fitting types:')
disp(['           No fitting: ' num2str(sum(fitting_type==0)) ' (' num2str(100*sum(fitting_type==0)/numel(fitting_type),3) '%)'])
disp(['               Newton: ' num2str(sum(fitting_type==1)) ' (' num2str(100*sum(fitting_type==1)/numel(fitting_type),3) '%)'])
disp(['      Mutant (random): ' num2str(sum(fitting_type==2)) ' (' num2str(100*sum(fitting_type==2)/numel(fitting_type),3) '%)'])
disp(['Epsilon (random move): ' num2str(sum(fitting_type==3)) ' (' num2str(100*sum(fitting_type==3)/numel(fitting_type),3) '%)'])
disp(' ')

% goodness=max(0.*cost_values,-diff([0,cost_values]))./cost_values;
% efficiency=goodness/sum(goodness);
% 
% disp('Efficency:')
% disp(['           No fitting: ' num2str(100*sum(efficiency(fitting_type==0)),2) '%'])
% disp(['               Newton: ' num2str(100*sum(efficiency(fitting_type==1)),2) '%'])
% disp(['      Mutant (random): ' num2str(100*sum(efficiency(fitting_type==2)),2) '%'])
% disp(['Epsilon (random move): ' num2str(100*sum(efficiency(fitting_type==3)),2) '%'])
% disp(' ')

%% Print brain
disp('Brain:')
% disp(['Parameter''s values:'])
for n=1:numel(parameters)
    disp(['p(' num2str(n) ') = ' num2str(parameters(n),10000)])
end

% disp('Neurons:')

for n=1:n_input_columns
    if log_distribution(n)==1
        disp(['y(' num2str(n) ') = ( log(input_' num2str(n) ') - ' num2str(min_data(n),1000) ' ) / ' num2str(range_data(n),1000)])
    else
        disp(['y(' num2str(n) ') = ( input_' num2str(n) ' - ' num2str(min_data(n),1000) ' ) / ' num2str(range_data(n),1000)])
    end
end

for neuron=neural_network_options.n_input_neurons+1:neural_network_options.n_neurons
    parent_neurons=find(...
        neural_network_options.layer_index==(neural_network_options.layer_index(neuron)-1)...
        );
    weight_indexes=...
        neural_network_options.weight_index_min(neuron):neural_network_options.weight_index_max(neuron);
    string_neuron_formula=['y(' num2str(neuron) ') = 1/(1+exp('];
    for k=1:numel(parent_neurons)
        string_neuron_formula=[string_neuron_formula...
            'y(' num2str(parent_neurons(k)) ') * p(' num2str(weight_indexes(k)) ') + '];
    end
    string_neuron_formula=[string_neuron_formula...
     'p(' num2str(neural_network_options.bias_index(neuron)) '))'];
    disp(string_neuron_formula)
end

neuron=n_neurons-n_objective_columns;
for n=n_input_columns+1:n_input_columns+n_objective_columns
    neuron=neuron+1;
    if log_distribution(n)==1
        disp(['output_' num2str(n-n_input_columns) ' = exp(y(' num2str(neuron) ') * ' num2str(range_data(n),1000) ' + ' num2str(min_data(n),1000) ')'])
    else
        disp(['output_' num2str(n-n_input_columns) ' = y(' num2str(neuron) ') * ' num2str(range_data(n),1000) ' + ' num2str(min_data(n),1000)])
    end
end

disp(' ')


%% Calulate secondary model outputs
if strcmp(secondary_data_file,data_file)==0
    secondary_raw_inputs=secondary_learning_data(:,1:n_input_columns);
    secondary_raw_objectives=secondary_learning_data(:,n_input_columns+1:n_input_columns+n_objective_columns);
    secondary_inputs=secondary_raw_inputs*NaN;
    secondary_objectives=secondary_raw_objectives*NaN;
    % Transform to 0-1
    for n=1:n_input_columns+n_objective_columns
        if n<=n_input_columns
            secondary_inputs(:,n)=transform_function(n,secondary_raw_inputs(:,n));
            
        else
            secondary_objectives(:,n-n_input_columns)=transform_function(n,secondary_raw_objectives(:,n-n_input_columns));
        end
    end
    [secondary_model_values,cost_global]=brain(neural_network_options,parameters,secondary_inputs,secondary_objectives);
    disp('Secondary data:')
    disp(['Secondary dataset MSE= ' num2str(cost_global)])
    secondary_raw_model_values=secondary_raw_objectives.*NaN;
    for n=1:n_objective_columns
        secondary_raw_model_values(:,n)=detransform_function(n+n_input_columns,secondary_model_values(:,n));
    end
    
    
    % Plot secondary approximations
    for column=1:size(raw_objectives,2)
        figure('units','normalized','outerposition',[0 0.5 0.5 0.5])
        hold on
        %     title(['Objetive ' num2str(column)])
        xlabel('Secondary data')
        ylabel('AI')
        data=secondary_raw_objectives(:,column);
        model=secondary_raw_model_values(:,column);
        
        valid=(abs(data)>0)&(abs(model)>0);
        average_uncert=mean(abs(data(valid)-model(valid))./data(valid));
        title(['Objetive' num2str(column) ' Av. uncert.=' num2str(average_uncert*100,3) '%'])
        
        plot(data,model,'.b')
        plot([min(data),max(data)],[min(data),max(data)],'--g')
        
        %     if log_distribution(n_input_columns+column)==1
        %         set(gca, 'YScale', 'log')
        %         set(gca, 'XScale', 'log')
        %     end
        box on
        grid on
end

end

diary off