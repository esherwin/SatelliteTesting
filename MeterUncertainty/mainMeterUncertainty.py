# Run meterUncertainty test case

import numpy as np
import matplotlib.pyplot as plt
import meterUncertainty


# Define input parameters
InputReleaseRate = 0
MeterAgeOption = 2021 # Possible inputs are 2016, 2018, and 2021 (or 0, 1, and 2, respectively)
PipeDiamOption = '2-inch' # Possible inputs are '2-inch', '4-inch', and '8-inch' (or 0, 1, and 2, respectively)
TestLocation = 'AZ' # Possible inputs are 'TX' and 'AZ' (or 0 and 1, respectively)
NumberMonteCarloDraws = 10000

ObservationStats, ObservationStatsNormed, ObservationRealizationHolder = meterUncertainty.meterUncertainty(InputReleaseRate, MeterAgeOption, PipeDiamOption, TestLocation, NumberMonteCarloDraws, hist=1, units='scfh')
print(ObservationStats, ObservationStatsNormed)

# def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
