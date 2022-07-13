# % Method of assigning uncertainty to controlled release meter tests
#
# % Adam R. Brandt (transcribed to Python by Evan D. Sherwin)
# % Department of Energy Resources Engineering
# % Stanford University

# % Script takes in average release rate over an X minute time period,
# % representing the flow that is going to be characterized by the team
# % making the measurement. Time period X varies by team according to how sensitive
# % the instrument is (how far away from the source it can see) and the
# % return time of the measurement.
#
# % We take as given the averaging period from the Rutherford script
#
# % Input parameters to function:
# InputReleaseRate, MeterAgeOption, PipeDiamOption, TestLocation, NumberMonteCarloDraws
# % 1. InputReleaseRate: for a particular release volume, the time averaged
# % volume flow rate in scfh of whole gas [scf per h or scfh]
#
# % 2. MeterAgeOption: Meter year such that 2016, 2018, 2021 = 0, 1, 2 (either format is acceptable)
#
# % 3. PipeDiamOption: Pipe diameter such that '2-inch' = 0, '4-inch' = 1, '8-inch' = 2
#
# % 4. TestLocation: Test location such that 'TX' = 0, 'AZ' = 1
#
# % 5. NumberMonteCarloDraws: Number of Monte Carlo Draws to perform
#
# % 6. hist: Binary variable indicating whether to produce a histograph
#
# % 7. units: 'scfh' or 'kgh' of CH4


# # Returns:
# ObservationStats: mean, 2.5, 5, 25, 50, 75, 95, 97.5 percentiles, stdev of simulated values
# ObservationStatsNormed: mean, 2.5, 5, 25, 50, 75, 95, 97.5 percentiles, stdev of simulated values divided by the mean
# ObservationRealizationHolder: Raw simulated values
# Currently outputs a histogram of results as well

# Required packages: pandas as pd, numpy as np, matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from UnitConversion import SCFH2kgh, mph2ms, gps2kgh, kgh2SCFH
import random as rnd

# % !!!HARD CODE THESE HERE BUT THESE WILL BE FUNCTION ARGUMENTS IN PRACTICE !!!
# InputReleaseRate = 10000
# MeterAgeOption = 2
# PipeDiamOption = 1
# TestLocation = 2
# NumberMonteCarloDraws = 1000

def meterUncertainty(InputReleaseRate, MeterOption, PipeDiamOption, TestLocation,field_recorded_mean,field_recorded_std,NumberMonteCarloDraws, hist=0, units='kgh'):
    # Rename parameters if in non-integer form
    if MeterOption == 162928:
        MeterAgeOption = 0
    elif MeterOption == 218645:
        MeterAgeOption = 1
    elif MeterOption == 308188:
        MeterAgeOption = 2
    elif MeterOption == 21175085:
        MeterAgeOption = 3

    if PipeDiamOption == 2:
        PipeDiamOption = 0
    elif PipeDiamOption == 4:
        PipeDiamOption = 1
    if PipeDiamOption == 8:
        PipeDiamOption = 2

    if TestLocation == 'TX':
        TestLocation = 0
    elif TestLocation == 'AZ':
        TestLocation = 1

    if MeterAgeOption <= 2:
        # % % Section on meter noise
    
        # % Table of fullscale flow lookup[scfh]
        # % Pipe diameters are 2, 4, and 8 inch schedule 40 pipe
        FullScaleFlowMeter1 = [27960, 106080, 416400]
        FullScaleFlowMeter2 = [27960, 106080, 416400]
        FullScaleFlowMeter3 = [22142, 84005, 329747]
        FullScaleFlowTable = np.array([FullScaleFlowMeter1, FullScaleFlowMeter2, FullScaleFlowMeter3])
    
        # % Look up the full scale flow for the meter and pipe combination in question
        FullScaleForObservation = FullScaleFlowTable[MeterAgeOption, PipeDiamOption]
    
        # % Fraction of full scale flow
        FractionOfFullScale = np.divide(InputReleaseRate, FullScaleForObservation)
    
        # % Table of uncertainty parameters for meter noise from Sierra
        # % See pp. 11 - 14 pf Sierra meter documentation: https://www.sierrainstruments.com/userfiles/file/datasheets/technical/640i-780i-datasheet.pdf?x=1184
    
        # % Qtherm gas calibration("8" calibration option for methane) is + / - 3 % of full scale
        # % Actual gas calibration("8A" calibration option for methane is + / - 0.75 %
        # % of reading at above 50 % of full scale, and 0.75 % of reading + 0.5 % of
        # % full scale at below 50 % of full scale
    
        # % Not clear in documentation, but we assume this is a 95 % CI, or 1.96 sigma uncertainty
        # % on a normally distributed error, so divide by 1.96 to get 1 sigma (SD)
        ErrorTermOfReading = np.array([0.00, 0.0075, 0.0075]) / 1.96
        ErrorTermOfFullScaleAbove50 = np.array([0.03, 0, 0]) / 1.96
        ErrorTermOfFullScaleBelow50 = np.array([0.03, 0.005, 0.005]) / 1.96
    
        if FractionOfFullScale > 0.5:
            ErrorTermOfFullScale = ErrorTermOfFullScaleAbove50
        else:
            ErrorTermOfFullScale = ErrorTermOfFullScaleBelow50
    
        NoiseTermSDTable = ErrorTermOfReading * InputReleaseRate + ErrorTermOfFullScale * FullScaleForObservation
        NoiseTermSD = NoiseTermSDTable[MeterAgeOption]

    # % % Section on meter bias
    #
    # % There can be consistent bias -- different from random meter noise -- due to meter
    # % mal-installation

    # % We use bias estimates from 12 different experiments comparing 2 different
    # % meters we ran on Oct 29-31 2021. See documentation for methods
    #
    # % Bias observations, in overall (Across whole time series) values for each
    # % meter divided by the mean of the two meter readings

    BiasObservations = np.array([0.943, 0.9866, 0.9982,  1.0273, 1.0043, 1.0017, 1.0034,
    1.0377,    0.999,    1.0308,    0.9726,    1.0339, 1.057,    1.0134,    1.0018,
    0.9727,    0.9957,    0.9983,    0.9966,    0.9623,    1.001,    0.9692,    1.0274,    0.9661])

    # % % Section on gas composition variability

    TXGasMoleFractions = np.array([86.2647,
    86.0076,
    86.5148,
    86.817,
    86.1962,
    85.7678,
    85.5685,
    85.6991,
    85.9681,
    86.227,
    85.3598,
    86.262,
    86.3767,
    86.5616,
    86.9023,
    85.3216,
    85.1271,
    84.7467,
    84.1816,
    84.8476,
    85.2905,
    85.2274,
    85.478,
    84.656,
    85.2363,
    88.2321,
    90.7401,
    85.0659,
    85.0633,
    85.0235]) / 100

    AZGasMoleFractions =np.array([96.27, 95.224, 96.125]) / 100

    if TestLocation:
        GasMoleFractionObservations = AZGasMoleFractions
    else:
        GasMoleFractionObservations = TXGasMoleFractions

    # % % Monte Carlo draw for the uncertainty on each observation

    # % Initialize a holder array to hold the results of monte carlo draws
    ObservationRealizationHolder = np.zeros(NumberMonteCarloDraws)

    # % Perform the monte carlo draws
    for ii in np.arange(1, NumberMonteCarloDraws):
        if MeterAgeOption <= 2:
            # % Draw the relevant parameters
            RealizedBiasValue = BiasObservations[np.random.randint(BiasObservations.size)] # % [fractional multipler]
            RealizedNoiseValue = np.random.normal(0, NoiseTermSD) # % [scfh]
            RealizedGasMoleFraction = GasMoleFractionObservations[np.random.randint(GasMoleFractionObservations.size)] # % [mol %]
            
            # If data was handwritten for the day we add an additional source of uncertainty
            rnd_norm = rnd.normalvariate(mu = field_recorded_mean, sigma = field_recorded_std)
            field_recorded_noise = InputReleaseRate * (1 - rnd_norm)
            
            # % Adjust for bias first by applying bias, then by applying noise, then
            # % by multiplying by the mole fraction of gas.Assume all are
            # % independent factors (e.g., methane mole fraction does not affect
            # meter bias)
            BiasAdjustedReleaseRate = InputReleaseRate * RealizedBiasValue
            BiasAndNoiseAdjustedReleaseRate =  BiasAdjustedReleaseRate + RealizedNoiseValue + field_recorded_noise
            ObservationRealization = BiasAndNoiseAdjustedReleaseRate * RealizedGasMoleFraction
    
            # % Retain the value in the loop into the holder array
            # % Estimated SCFH of actual methane
            ObservationRealizationHolder[ii] = ObservationRealization
        else:
            # Coriolis uncertainty
            RealizedBiasValue = 1
            RealizedNoiseValue = (0 if InputReleaseRate == 0 else 316.92 * InputReleaseRate ** -0.969)
            RealizedNoiseValue = (RealizedNoiseValue/100)*InputReleaseRate
            
            RealizedGasMoleFraction = GasMoleFractionObservations[np.random.randint(GasMoleFractionObservations.size)] # % [mol %]
            
            # If data was handwritten for the day we add an additional source of uncertainty
            rnd_norm = rnd.normalvariate(mu = field_recorded_mean, sigma = field_recorded_std)
            field_recorded_noise = InputReleaseRate * (1 - rnd_norm)
            
            # % Adjust for bias first by applying bias, then by applying noise, then
            # % by multiplying by the mole fraction of gas.Assume all are
            # % independent factors (e.g., methane mole fraction does not affect
            # meter bias)
            BiasAdjustedReleaseRate = InputReleaseRate * RealizedBiasValue
            BiasAndNoiseAdjustedReleaseRate =  BiasAdjustedReleaseRate + RealizedNoiseValue + field_recorded_noise
            ObservationRealization = BiasAndNoiseAdjustedReleaseRate * RealizedGasMoleFraction
    
            # % Retain the value in the loop into the holder array
            # % Estimated SCFH of actual methane
            ObservationRealizationHolder[ii] = ObservationRealization
            
            
    # Convert units to kgh if specified
    if units=='kgh':
        # Use Unit Conversion Script
        ObservationRealizationHolder = SCFH2kgh(ObservationRealizationHolder, T=21.1)

    # Plot histogram if desired
    if hist:
        plt.hist(ObservationRealizationHolder, bins=50)
        plt.show()

    if InputReleaseRate == 0:
        ObservationStats = np.zeros(3)
    else:
        ObservationStats =np.array([np.mean(ObservationRealizationHolder),
        np.percentile(ObservationRealizationHolder, 2.5),
        # np.percentile(ObservationRealizationHolder, 5),
        # np.percentile(ObservationRealizationHolder, 25),
        # np.percentile(ObservationRealizationHolder, 50),
        # np.percentile(ObservationRealizationHolder, 75),
        # np.percentile(ObservationRealizationHolder, 95),
        np.percentile(ObservationRealizationHolder, 97.5)]) #,
        # np.std(ObservationRealizationHolder)])

    if InputReleaseRate == 0:
        ObservationStatsNormed = np.zeros(3)
    else:
        ObservationStatsNormed =np.array([np.mean(ObservationRealizationHolder),
        np.percentile(ObservationRealizationHolder, 2.5),
        # np.percentile(ObservationRealizationHolder, 5),
        # np.percentile(ObservationRealizationHolder, 25),
        # np.percentile(ObservationRealizationHolder, 50),
        # np.percentile(ObservationRealizationHolder, 75),
        # np.percentile(ObservationRealizationHolder, 95),
        np.percentile(ObservationRealizationHolder, 97.5)])/ np.mean(ObservationRealizationHolder) #,
        # np.std(ObservationRealizationHolder)]) / np.mean(ObservationRealizationHolder)

    return ObservationStats, ObservationStatsNormed, ObservationRealizationHolder