mdl = 'Topo_5Gsimulink_handover';
load_system(mdl);

% scenario parameter {rsrp_amp, AMF_state, gNB1_status, gNB2_status, tag}
cases = {
    {30, 1, 1, 1, 'baseline'}      % Normal operation + switching
    {30, 1, 0, 1, 'gNB1_down'}     
    {30, 0, 1, 1, 'AMF_reject'}    
    {50, 1, 1, 1, 'no_handover'}  
};

for k = 1:numel(cases)
    cfg = cases{k};
    amp           = cfg{1};
    amf_state     = cfg{2};
    gnb1_status   = cfg{3};
    gnb2_status   = cfg{4};
    tag           = cfg{5};

    % variables write in（Simulink From Workspace / Constant）
    assignin('base','amp',amp);
    assignin('base','amf_state',amf_state);
    assignin('base','gnb1_status',gnb1_status);
    assignin('base','gnb2_status',gnb2_status);

    simOut = sim(mdl,'StopTime','40','SaveOutput','off');

    %get 'To Workspace' signal
    disp('------ DEBUG -------')
    disp(['Scene  ', tag])
    %disp(['cs1 10 = ', mat2str(cs1(1:10)')])
    %disp(['best 10 = ', mat2str(best(1:10)')])
    disp('--------------------')

    % Saved as .mat files
    save(['result_' tag '.mat'],'cs1','cs2','req1','req2','best','AMF');
    fprintf('Scene %-12s  done.\n',tag);
end
load result_baseline.mat
plot(cs1)            
plot(best)          

