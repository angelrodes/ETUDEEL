function [ output_args ] = plot_brain( neural_network_options,parameters )
% PLOT_BRAIN Plots the brain structure
%  Angel Rodes, 2022

% clear
hold off
plot(0,-max(neural_network_options.layer_structure)/2-1,'.w')
hold on
plot(max(neural_network_options.layer_index)+1,+max(neural_network_options.layer_structure)/2+1,'.w')

text(0.5,0,[char(8594)])
text(max(neural_network_options.layer_index)+0.5,0,[char(8594)])

layer_index=neural_network_options.layer_index;
x=layer_index;
y=x.*NaN;
h_in_layer=0;
for neuron=1:neural_network_options.n_neurons
    h_in_layer=h_in_layer+1;
    if neuron>1
        if layer_index(neuron)~=layer_index(neuron-1)
            h_in_layer=1;
        end
    end
    y(neuron)=h_in_layer-1-(neural_network_options.layer_structure(layer_index(neuron))-1)/2;
   
    if neuron>neural_network_options.n_input_neurons
        
        % plot weights
        parent_neurons=find(...
            neural_network_options.layer_index==(neural_network_options.layer_index(neuron)-1));
        weight_indexes=...
            neural_network_options.weight_index_min(neuron):neural_network_options.weight_index_max(neuron);
        for n=1:numel(weight_indexes)
            thickness=min(3,abs(parameters(weight_indexes(n))));
            if thickness>0.1
                if parameters(weight_indexes(n))>0
                    plot([x(parent_neurons(n)),x(neuron)],[y(parent_neurons(n)),y(neuron)],'-b','LineWidth',thickness)
                else
                    plot([x(parent_neurons(n)),x(neuron)],[y(parent_neurons(n)),y(neuron)],'-r','LineWidth',thickness)
                end
            end
        end
        
        % plot bias
        %  r=min(1,abs(parameters(neural_network_options.bias_index(neuron))))*0.5;
        r=0.3;
        thickness=min(3,abs(parameters(neural_network_options.bias_index(neuron))));
        th = 0:pi/10:2*pi;
        xunit = r * cos(th) + x(neuron);
        yunit = r * sin(th) + y(neuron);
        if thickness>0.1
            if parameters(neural_network_options.bias_index(neuron))>0
                plot(xunit, yunit,'-b','LineWidth',thickness)
            else
                plot(xunit, yunit,'-r','LineWidth',thickness)
            end
        end
    end
        
%      text(x(neuron),y(neuron),['y' num2str(neuron)],'Color','k','HorizontalAlignment', 'center')
end

for neuron=1:neural_network_options.n_neurons
    text(x(neuron),y(neuron),['y' num2str(neuron)],...
        'Color','k','HorizontalAlignment', 'center',...
        'BackgroundColor','w')
end

end

