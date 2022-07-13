import plotFunc
import pandas as pd

cmData = pd.read_csv('/Users/evansherwin/PycharmProjects/SatelliteTesting/Controlled_Release_2021-main/matchedDF_CarbonMapper_1sigma.csv')
# cmData = pd.read_excel('/Users/evansherwin/PycharmProjects/SatelliteTesting/Controlled_Release_2021-main/Sent_to_teams/Carbon Mapper/matchedDF_CarbonMapper_unblindedToCM.xlsx',
#                        sheet_name="matchedDF_CarbonMapper_unblinde", engine='openpyxl')

plotFunc.plotMain(cmData, cmData, cmData)