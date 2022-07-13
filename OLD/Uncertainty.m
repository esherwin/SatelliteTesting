% Method of assigning uncertainty to controlled release meter tests

% Adam R. Brandt
% Department of Energy Resources Engineering
% Stanford University

close all;
clear all;
clc;


% Script takes in average release rate over an X minute time period,
% representing the flow that is going to be characterized by the team
% making the measurement. Time period X varies by team according to how sensitive
% the instrument is (how far away from the source it can see) and the
% return time of the measurement.

% We take as given the averaging period from the Rutherford script

% Input parameters to function:

% 1. Input release rate: for a particular release volume, the time averaged
% volume flow rate in scfh of whole gas [scf per h or scfh]

% 2. Meter age: Meter year such that 1, 2, 3 = 2016, 2018, 2021

% 3. PipeDiam: Pipe diameter such that 1 = 2 inch, 2 = 4 inch, 3 = 8 inch

% 4. Test Location: Test location such that TX = 1, AZ = 2

% 5. Number of Monte Carlo Draws to perform


% !!!HARD CODE THESE HERE BUT THESE WILL BE FUNCTION ARGUMENTS IN PRACTICE !!!
InputReleaseRate = 10000;
MeterAgeOption = 3;
PipeDiamOption = 2;
TestLocation = 1;
NumberMonteCarloDraws = 10000;


%% Section on meter noise

% Table of fullscale flow lookup [scfh]
% Pipe diameters are 2, 4, and 8 inch schedule 40 pipe
FullScaleFlowMeter1 = [27960; 106080; 416400];
FullScaleFlowMeter2 = [27960; 106080; 416400];
FullScaleFlowMeter3 = [22142; 84005; 329747];
FullScaleFlowTable = [FullScaleFlowMeter1, FullScaleFlowMeter2, FullScaleFlowMeter3];

% Look up the full scale flow for the meter and pipe combination in
% question
FullScaleForObservation = FullScaleFlowTable(PipeDiamOption, MeterAgeOption);

% Fraction of full scale flow
FractionOfFullScale = InputReleaseRate/FullScaleForObservation;


% Table of uncertainty parameters for meter noise from Sierra
% See pp. 11-14 pf Sierra meter documentation: https://www.sierrainstruments.com/userfiles/file/datasheets/technical/640i-780i-datasheet.pdf?x=1184

% Qtherm gas calibration ("8" calibration option for methane) is +/- 3% of full scale
% Actual gas calibration ("8A" calibration option for methane is +/- 0.75%
% of reading at above 50% of full scale, and 0.75% of reading + 0.5% of
% full scale at below 50% of full scale

% Not clear in documentation, but we assume this is a 95% CI, or 1.96 sigma uncertainty
% on a normally distributed error, so divide by 1.96 to get 1 sigma (SD)
ErrorTermOfReading = [0.00 0.0075 0.0075]/1.96;
ErrorTermOfFullScaleAbove50 = [0.03 0 0]/1.96;
ErrorTermOfFullScaleBelow50 = [0.03 0.005 0.005]/1.96;

if FractionOfFullScale > 0.5
    ErrorTermOfFullScale = ErrorTermOfFullScaleAbove50;
else
    ErrorTermOfFullScale = ErrorTermOfFullScaleBelow50;
end

NoiseTermSDTable = ErrorTermOfReading*InputReleaseRate + ErrorTermOfFullScale*FullScaleForObservation;
NoiseTermSD = NoiseTermSDTable(MeterAgeOption);



%% Section on meter bias

% There can be consistent bias -- different from random meter noise -- due to meter
% mal-installation

% We use bias estimates from 12 different experiments comparing 2 different
% meters we ran on Oct 29-31 2021. See documentation for methods


% Bias observations, in overall (Across whole time series) values for each
% meter divided by the mean of the two meter readings

BiasObservations = [0.943	0.9866	0.9982	1.0273	1.0043	1.0017	1.0034	...
    1.0377	0.999	1.0308	0.9726	1.0339 1.057	1.0134	1.0018	...
    0.9727	0.9957	0.9983	0.9966	0.9623	1.001	0.9692	1.0274	0.9661];



%% Section on gas composition variability

TXGasMoleFractions =[86.2647
86.0076
86.5148
86.817
86.1962
85.7678
85.5685
85.6991
85.9681
86.227
85.3598
86.262
86.3767
86.5616
86.9023
85.3216
85.1271
84.7467
84.1816
84.8476
85.2905
85.2274
85.478
84.656
85.2363
88.2321
90.7401
85.0659
85.0633
85.0235]/100;

AZGasMoleFractions = [96.27 95.224 96.125]/100;


if TestLocation == 1
    GasMoleFractionObservations = TXGasMoleFractions;
else
    GasMoleFractionObservations = AZGasMoleFractions;
end



%% Monte Carlo draw for the uncertainty on each observation

% Initialize a holder array to hold the results of monte carlo draws
ObservationRealizationHolder = zeros(1,NumberMonteCarloDraws);

% Perform the monte carlo draws
for ii = 1:NumberMonteCarloDraws
    % Draw the relevant parameters
    RealizedBiasValue = BiasObservations(randi([1 size(BiasObservations,2)])); %[fractional multipler]
    RealizedNoiseValue = normrnd(0,NoiseTermSD); %[scfh]
    RealizedGasMoleFraction = GasMoleFractionObservations(randi([1 size(GasMoleFractionObservations,1)])); % [mol%]
    
    % Adjust for bias first by applying bias, then by applying noise, then
    % by multiplying by the mole fraction of gas. Assume all are
    % independent factors (e.g., methane mole fraction does not affect
    % meter bias)
    BiasAdjustedReleaseRate = InputReleaseRate*RealizedBiasValue;
    BiasAndNoiseAdjustedReleaseRate =  BiasAdjustedReleaseRate + RealizedNoiseValue;
    ObservationRealization = BiasAndNoiseAdjustedReleaseRate*RealizedGasMoleFraction;

    % Retain the value in the loop into the holder array
    % Estimated SCFH of actual methane
    ObservationRealizationHolder(ii) = ObservationRealization;

end


figure
histogram(ObservationRealizationHolder)


ObservationStats = [mean(ObservationRealizationHolder);...
    prctile(ObservationRealizationHolder,2.5);...
    prctile(ObservationRealizationHolder,5);...
    prctile(ObservationRealizationHolder,25);...
    prctile(ObservationRealizationHolder,50);...
    prctile(ObservationRealizationHolder,75);...
    prctile(ObservationRealizationHolder,95);...
    prctile(ObservationRealizationHolder,97.5);...
    std(ObservationRealizationHolder)
    ]

ObservationStatsNormed = [mean(ObservationRealizationHolder);...
    prctile(ObservationRealizationHolder,2.5);...
    prctile(ObservationRealizationHolder,5);...
    prctile(ObservationRealizationHolder,25);...
    prctile(ObservationRealizationHolder,50);...
    prctile(ObservationRealizationHolder,75);...
    prctile(ObservationRealizationHolder,95);...
    prctile(ObservationRealizationHolder,97.5);...
    std(ObservationRealizationHolder)
    ]/mean(ObservationRealizationHolder)





