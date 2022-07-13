# imports
import datetime
import pytz
import math
import os.path
import pathlib
import pandas as pd
import bisect
import numpy as np
from UnitConversion import convertUnits, SCFH2kgh, mph2ms, applyComposition, gps2kgh, kgh2SCFH, gps2scfh

def loaddata():
    """Load all data from testing"""
    
    cwd = os.getcwd()    
    
    # load Bridger data 
    DataPath = os.path.join(cwd, 'BridgerTestData')    

    print("Loading Bridger Stage 1 (HRRR) data ...")
    bridgerHRRR_path = os.path.join(DataPath, 'XOM0011 Stanford CR - HRRR.xlsx')
    timestamp_path = os.path.join(DataPath,'Bridger_Timestamps.csv')
    bridgerHRRRDF = loadBridgerData(bridgerHRRR_path, timestamp_path)
    bridgerHRRRDF['WindType'] = 'HRRR'
    bridgerHRRRDF['OperatorSet'] = 'Bridger'
    bridgerHRRRDF = bridgerHRRRDF[bridgerHRRRDF['EquipmentUnitID'] == 33931]
    bridgerHRRRDF['UnblindingStage'] = 1

    print("Loading Bridger Stage 1 (NAM12) data ...")
    bridgerNAM12_path = os.path.join(DataPath, 'XOM0011 Stanford CR - NAM12.xlsx')
    timestamp_path = os.path.join(DataPath,'Bridger_Timestamps.csv')
    bridgerNAM12DF = loadBridgerData(bridgerNAM12_path, timestamp_path)
    bridgerNAM12DF['WindType'] = 'NAM12'
    bridgerNAM12DF['OperatorSet'] = 'Bridger'
    bridgerNAM12DF = bridgerNAM12DF[bridgerNAM12DF['EquipmentUnitID'] == 33931]
    bridgerNAM12DF['UnblindingStage'] = 1

    print("Loading Bridger Stage 2 data ...")
    bridgerSonic_path = os.path.join(DataPath, 'XOM0011 Stanford CR - Anemometer.xlsx')
    timestamp_path = os.path.join(DataPath,'Bridger_Timestamps.csv')
    bridgerSonicDF = loadBridgerData(bridgerSonic_path, timestamp_path)
    bridgerSonicDF['WindType'] = 'Sonic'
    bridgerSonicDF['OperatorSet'] = 'Bridger'
    bridgerSonicDF = bridgerSonicDF[bridgerSonicDF['EquipmentUnitID'] == 33931]
    bridgerSonicDF['UnblindingStage'] = 2

    print("Loading Bridger Stage 3 data ...")
    bridgerSonic_path = os.path.join(DataPath, 'XOM0011 Stanford CR - Stage 3.xlsx')
    timestamp_path = os.path.join(DataPath,'Bridger_Timestamps.csv')
    bridgerR3DF = loadBridgerData(bridgerSonic_path, timestamp_path)
    bridgerR3DF['WindType'] = 'Sonic'
    bridgerR3DF['OperatorSet'] = 'Bridger'
    bridgerR3DF = bridgerR3DF[bridgerR3DF['EquipmentUnitID'] == 33931]
    bridgerR3DF['UnblindingStage'] = 3

    # append Bridger data into single DF
    bridgerDF = pd.concat([bridgerHRRRDF, bridgerNAM12DF, bridgerSonicDF], ignore_index=True)
  
    # Delete rows where Bridger passed over before Stanford was prepared to release
    date_cutoff = pd.to_datetime('2021.11.03 17:38:06')
    date_cutoff = date_cutoff.tz_localize('UTC')
    bridgerDF = bridgerDF.drop(bridgerDF[(bridgerDF['Operator_Timestamp'] < date_cutoff)].index)
    #bridgerDF = bridgerDF.drop(bridgerDF.index[[0,1,116,117]])
    bridgerDF = bridgerDF.reset_index()    


    # load Carbon Mapper data
    DataPath = os.path.join(cwd, 'CarbonMapperTestData')    

    print("Loading Carbon Mapper Stage 1 data...")
    CarbonMapper_path = os.path.join(DataPath, 'CarbonMapper_ControlledRelease_submission.csv')
    timestamp_path = os.path.join(DataPath,'CarbonMapper_Timestamps.csv')    
    CarbonMapperR1DF = loadCarbonMapperData(CarbonMapper_path, timestamp_path)
    CarbonMapperR1DF['WindType'] = 'HRRR'
    CarbonMapperR1DF['OperatorSet'] = 'CarbonMapper'
    CarbonMapperR1DF['UnblindingStage'] = 1 
    
    print("Loading Carbon Mapper Stage 2 data...")
    CarbonMapper_path = os.path.join(DataPath, 'CarbonMapper_ControlledRelease_submission_PostPhase1.csv')
    timestamp_path = os.path.join(DataPath,'CarbonMapper_Timestamps.csv') 
    CarbonMapperR2DF = loadCarbonMapperData(CarbonMapper_path, timestamp_path)
    CarbonMapperR2DF['WindType'] = 'Sonic'
    CarbonMapperR2DF['OperatorSet'] = 'CarbonMapper' 
    CarbonMapperR2DF['UnblindingStage'] = 2 

    print("Loading Carbon Mapper Stage 3 data...")
    CarbonMapper_path = os.path.join(DataPath, 'CarbonMapper_ControlledRelease_submission_Phase3.csv')
    timestamp_path = os.path.join(DataPath,'CarbonMapper_Timestamps.csv') 
    CarbonMapperR3DF = loadCarbonMapperData(CarbonMapper_path, timestamp_path)
    CarbonMapperR3DF['WindType'] = 'Sonic'
    CarbonMapperR3DF['OperatorSet'] = 'CarbonMapper' 
    CarbonMapperR3DF['UnblindingStage'] = 3 
    
    CarbonMapperDF = pd.concat([CarbonMapperR1DF, CarbonMapperR2DF, CarbonMapperR3DF], ignore_index=True)

    # Delete rows where Carbon Mapper passed over before Stanford was prepared to release
    date_cutoff = pd.to_datetime('2021.07.30 15:32:00')
    date_cutoff = date_cutoff.tz_localize('UTC')
    CarbonMapperDF = CarbonMapperDF.drop(CarbonMapperDF[(CarbonMapperDF['Operator_Timestamp'] < date_cutoff)].index)
    #bridgerDF = bridgerDF.drop(bridgerDF.index[[0,1,116,117]])
    CarbonMapperDF = CarbonMapperDF.reset_index()    


    # load GHGSat data 
    DataPath = os.path.join(cwd, 'GHGSatTestData')   
    
    print("Loading GHGSat Stage 1 data...")
    GHGSat_path = os.path.join(DataPath, 'GHG-1496-6006-a  AV1 Stanford Controlled Release Data Report.csv')
    timestamp_path = os.path.join(DataPath,'GHGSat_Timestamps.csv') 
    GHGSatR1DF = loadGHGSatData(GHGSat_path, timestamp_path)
    GHGSatR1DF['WindType'] = 'NASA-GEOS'
    GHGSatR1DF['OperatorSet'] = 'GHGSat'
    GHGSatR1DF['UnblindingStage'] = 1    
    GHGSatR1DF = GHGSatR1DF.drop(GHGSatR1DF[(GHGSatR1DF['EquipmentUnitID'] == 2)].index)

    print("Loading GHGSat Stage 2 data...")
    GHGSat_path = os.path.join(DataPath, 'GHG-1496-6006-a  AV1 Stanford Controlled Release Data Report_Stage2.csv')
    timestamp_path = os.path.join(DataPath,'GHGSat_Timestamps.csv') 
    GHGSatR2DF = loadGHGSatData(GHGSat_path, timestamp_path)
    GHGSatR2DF['WindType'] = 'Sonic'
    GHGSatR2DF['OperatorSet'] = 'GHGSat'
    GHGSatR2DF['UnblindingStage'] = 2    
    GHGSatR2DF = GHGSatR2DF.drop(GHGSatR2DF[(GHGSatR2DF['EquipmentUnitID'] == 2)].index)

    print("Loading GHGSat Stage 3 data...")
    GHGSat_path = os.path.join(DataPath, 'GHG-1496-6006-a  AV1 Stanford Controlled Release Data Report_Stage3.csv')
    timestamp_path = os.path.join(DataPath,'GHGSat_Timestamps.csv') 
    GHGSatR3DF = loadGHGSatData(GHGSat_path, timestamp_path)
    GHGSatR3DF['WindType'] = 'Sonic'
    GHGSatR3DF['OperatorSet'] = 'GHGSat'
    GHGSatR3DF['UnblindingStage'] = 3
    GHGSatR3DF = GHGSatR3DF.loc[:, ~GHGSatR3DF.columns.str.contains('^Unnamed')]    
    GHGSatR3DF = GHGSatR3DF.drop(GHGSatR3DF[(GHGSatR3DF['EquipmentUnitID'] == 2)].index)
    
    GHGSatDF = pd.concat([GHGSatR1DF, GHGSatR2DF, GHGSatR3DF], ignore_index=True)

    
    operatorDF = pd.concat([bridgerDF, CarbonMapperDF, GHGSatDF], ignore_index=True)    
    
  
    # load Bridger quadratherm data
    print("Loading Bridger Quadratherm data...")
    DataPath = os.path.join(cwd, 'BridgerTestData') 
    meterDF_Bridger = loadMeterData_Bridger(DataPath)
    
    # load Carbon Mapper quadratherm data
    print("Loading Carbon Mapper Quadratherm data...")
    DataPath = os.path.join(cwd, 'CarbonMapperTestData')  
    meterDF_CarbonMapper = loadMeterData_CarbonMapper(DataPath)
    
    # load GHGSat quadratherm data
    print("Loading GHGSat Quadratherm data...")
    DataPath = os.path.join(cwd, 'GHGSatTestData')  
    meterDF_GHGSat = loadMeterData_GHGSat(DataPath)
    
    meterDF_All = pd.concat([meterDF_Bridger, meterDF_CarbonMapper, meterDF_GHGSat])
    
    
    DataPath = os.path.join(cwd, 'BridgerTestData')  
    # load Bridger anemometer data
    print("Loading anemometer data...")
    sonic_path = os.path.join(DataPath, 'Sonic\\')
    sonicDF_Bridger = combineAnemometer_Bridger(sonic_path)
    sonicDF_Bridger['OperatorSet'] = 'Bridger'
    
    DataPath = os.path.join(cwd, 'CarbonMapperTestData')  
    # load Carbon Mapper anemometer data
    print("Loading anemometer data...")
    sonic_path = os.path.join(DataPath, 'Sonic\\')
    sonicDF_CarbonMapper = combineAnemometer_CarbonMapper(sonic_path)
    sonicDF_CarbonMapper['OperatorSet'] = 'CarbonMapper'
    
    DataPath = os.path.join(cwd, 'GHGSatTestData')  
    # load GHGSat anemometer data
    print("Loading anemometer data...")
    sonic_path = os.path.join(DataPath, 'Sonic\\')
    sonicDF_GHGSat = combineAnemometer_GHGSat(sonic_path)
    sonicDF_GHGSat['OperatorSet'] = 'GHGSat'
    sonicDF_All = pd.concat([sonicDF_Bridger, sonicDF_CarbonMapper, sonicDF_GHGSat])
    
    return operatorDF, meterDF_All, sonicDF_All
    
def loadBridgerData(filepath, timestamp_path):
    """Load bridger data from report and format datetimes."""
    dfraw = pd.read_excel(filepath, sheet_name='emitter_group_scan', skiprows=4, engine='openpyxl')
    # convert datetime data to a datetime object; format: 04-Oct-2021 18:34:31
    dfraw['Flight Feature Time (UTC)'] = dfraw.apply(
        lambda x: datetime.datetime.strptime(x['Flight Feature Time (UTC)'], '%d-%b-%Y %H:%M:%S'), axis=1)
    dfraw['Flight Feature Time (UTC)'] = dfraw.apply(
        lambda x: x['Flight Feature Time (UTC)'].replace(tzinfo=pytz.timezone("UTC")), axis=1)

    dfraw['Detection Time (UTC)'] = dfraw.apply(
        lambda x: pd.NA if pd.isna(x['Detection Time (UTC)']) else
        datetime.datetime.strptime(x['Detection Time (UTC)'], '%d-%b-%Y %H:%M:%S'), axis=1)
    dfraw['Detection Time (UTC)'] = dfraw.apply(
        lambda x: pd.NA if pd.isna(x['Detection Time (UTC)']) else
        x['Detection Time (UTC)'].replace(tzinfo=pytz.timezone("UTC")), axis=1)

    dfraw.loc[dfraw["Detection Time (UTC)"].isnull(),'Detection Time (UTC)'] = dfraw["Flight Feature Time (UTC)"]
    

    column_names = [
        "PerformerExperimentID",
        "FacilityID",
        "EquipmentUnitID",
        "DateOfSurvey",
        "Timestamp (hyperspectral technologies only)",
        "StartTime",
        "EndTime",
        "SurveyTime",
        "Gas",
        "PlumeLength (hyperspectral technologies only)",
        "FacilityEmissionRate",
        "FacilityEmissionRateUpper",
        "FacilityEmissionRateLower",
        "UncertaintyType",
        "WindSpeed",
        "WindDirection",
        "TransitDirection",
        "QC filter",
        "NumberOfEmissionSourcesReported"]

    df = pd.DataFrame(columns = column_names)

    df["Timestamp (hyperspectral technologies only)"] = dfraw["Detection Time (UTC)"]
    df["Gas"] = 'Methane'
    df["FacilityEmissionRate"] = SCFH2kgh(dfraw['Emission Rate (SCFH)'], T=20)
    df["WindSpeed"] = mph2ms(dfraw["Detection Wind Speed (mph)"])
    df["EquipmentUnitID"] = dfraw["Emission Location Id"]
    
    df.rename(columns={'Timestamp (hyperspectral technologies only)':'Operator_Timestamp'}, inplace=True)
    df['Operator_Timestamp'] = pd.to_datetime(df['Operator_Timestamp'])
    
    # Bridger reported additional rows for emissions from the Rawhide trailer. Select only rows where emission Location
    # ID = 33931 (the release point) and ignore rows where emission point is the leaky trailer
    df = df.loc[df['EquipmentUnitID'] == 33931] 
    
    StanfordTimestamps = pd.read_csv(timestamp_path, header = None, names = ['Stanford_timestamp'], parse_dates=True)
    StanfordTimestamps['Stanford_timestamp'] = pd.to_datetime(StanfordTimestamps['Stanford_timestamp'])
    StanfordTimestamps['Stanford_timestamp'] = StanfordTimestamps.apply(
        lambda x: x['Stanford_timestamp'].replace(tzinfo=pytz.timezone("UTC")), axis=1)
    #StanfordTimestamps.set_index('Stanford_timestamp', inplace = True)
    
    tol = pd.Timedelta('1 minute')
    df = pd.merge_asof(left=df.sort_values('Operator_Timestamp'),right=StanfordTimestamps.sort_values('Stanford_timestamp'), right_on='Stanford_timestamp',left_on='Operator_Timestamp',direction='nearest',tolerance=tol)     
    df.loc[df['Stanford_timestamp'].isnull(),'Stanford_timestamp'] = df["Operator_Timestamp"]
    
    return df

def loadCarbonMapperData(filepath, timestamp_path):
    """Load Carbon Mapper data from report and format datetimes."""
    
    #df = pd.read_excel(filepath, sheet_name='Survey Summary', skiprows=0, engine='openpyxl')
    df = pd.read_csv(filepath, parse_dates=[['DateOfSurvey', 'Timestamp (hyperspectral technologies only)']])

    df.rename(columns={'DateOfSurvey_Timestamp (hyperspectral technologies only)':'Operator_Timestamp'}, inplace=True)

    # Carbon Mapper reports a QC_filter for all passes, including non-retrievals
    # Therfore, filter out blank rows in the spreadsheet by removing rows with 
    # null QC_filter
    # (Jeff - Need to verify)
    
    # QC filter = 1 was manually added to Carbon Mapper for rounds 2 and 3
    
    df['Operator_Timestamp'].fillna(value=np.nan, inplace=True)
    df = df[df["QC filter"].notnull()]
    
    
    df['Operator_Timestamp'] = df.apply(
        lambda x: pd.NA if pd.isna(x['Operator_Timestamp']) else
        datetime.datetime.strptime(x['Operator_Timestamp'], '%m/%d/%Y %H:%M:%S'), axis=1)

    df['Operator_Timestamp'] = df.apply(
        lambda x: pd.NA if pd.isna(x['Operator_Timestamp']) else
        x['Operator_Timestamp'].replace(tzinfo=pytz.timezone("US/Central")), axis=1)


    df['Operator_Timestamp'] = df['Operator_Timestamp'].apply(lambda x: x.astimezone(pytz.timezone('UTC')))
    
    StanfordTimestamps = pd.read_csv(timestamp_path, header = None, names = ['Stanford_timestamp'], parse_dates=True)
    StanfordTimestamps['Stanford_timestamp'] = pd.to_datetime(StanfordTimestamps['Stanford_timestamp'])
    StanfordTimestamps['Stanford_timestamp'] = StanfordTimestamps.apply(
        lambda x: x['Stanford_timestamp'].replace(tzinfo=pytz.timezone("US/Central")), axis=1)
    StanfordTimestamps['Stanford_timestamp'] = StanfordTimestamps['Stanford_timestamp'].apply(lambda x: x.astimezone(pytz.timezone('UTC')))
    #StanfordTimestamps.set_index('Stanford_timestamp', inplace = True)
    
    tol = pd.Timedelta('1 minute')
    df = pd.merge_asof(left=df.sort_values('Operator_Timestamp'),right=StanfordTimestamps.sort_values('Stanford_timestamp'), right_on='Stanford_timestamp',left_on='Operator_Timestamp',direction='nearest',tolerance=tol)     
    df.loc[df['Stanford_timestamp'].isnull(),'Stanford_timestamp'] = df["Operator_Timestamp"]
    
    
    return df

def loadGHGSatData(filepath, timestamp_path):
    """Load GHGSat data from report and format datetimes."""
    #df = pd.read_excel(filepath, sheet_name='Survey Summary', skiprows=0, engine='openpyxl')
    df = pd.read_csv(filepath, parse_dates=[['DateOfSurvey', 'Timestamp (hyperspectral technologies only)']])
    
    df.rename(columns={'DateOfSurvey_Timestamp (hyperspectral technologies only)':'Operator_Timestamp'}, inplace=True)
    cwd = os.getcwd()   
    QC_filter = pd.read_csv(os.path.join(cwd, 'GHGSatTestData','QC_filter.csv'), header = None, names = ['QC filter'])  
        
    # GHGSat does not report a timestamp for all passes. Non-retrievals are identified
    # by rows without a timestamp.
    # Therfore, need to choose a different column to filter out blank rows in the 
    # spreadsheet. Choose to filter out  rows with a blank Performer Experiment ID
    # (Jeff - Need to verify this is a good approacah)
    
    df = df[df["PerformerExperimentID"].notnull()]
    df['QC filter'] = QC_filter['QC filter']

    # TEMPORARY: Remove the following rows with failed retrievals:
        # 	PerformerExperimentID = 1496-1-302-827-1026-4
        #  	PerformerExperimentID = 1496-2-305-537-736-51
        # 	PerformerExperimentID = 1496-4-102-1292-1491-118
        # 	PerformerExperimentID = 1496-4-123-527-726-138
        # 	PerformerExperimentID = 1496-4-129-863-1062-144
        # 	PerformerExperimentID = 1496-4-141-613-812-154

    df = df.drop(df[(df['PerformerExperimentID'] == '1496-1-302-827-1026-4')].index)
    df = df.drop(df[(df['PerformerExperimentID'] == '1496-2-305-537-736-51')].index)
    df = df.drop(df[(df['PerformerExperimentID'] == '1496-4-102-1292-1491-118')].index)
    df = df.drop(df[(df['PerformerExperimentID'] == '1496-4-123-527-726-138')].index)
    df = df.drop(df[(df['PerformerExperimentID'] == '1496-4-129-863-1062-144')].index)
    df = df.drop(df[(df['PerformerExperimentID'] == '1496-4-141-613-812-154')].index)

    df['Operator_Timestamp'] = df.apply(
        lambda x: pd.NA if pd.isna(x['Operator_Timestamp']) else
        datetime.datetime.strptime(x['Operator_Timestamp'], '%Y-%m-%d %H:%M:%S'), axis=1)    
  
    df['Operator_Timestamp'] = df.apply(
        lambda x: pd.NA if pd.isna(x['Operator_Timestamp']) else
        x['Operator_Timestamp'].replace(tzinfo=pytz.timezone("UTC")), axis=1)

    StanfordTimestamps = pd.read_csv(timestamp_path, header = None, names = ['Stanford_timestamp'], parse_dates=True)
    StanfordTimestamps['Stanford_timestamp'] = pd.to_datetime(StanfordTimestamps['Stanford_timestamp'])
    StanfordTimestamps['Stanford_timestamp'] = StanfordTimestamps.apply(
        lambda x: x['Stanford_timestamp'].replace(tzinfo=pytz.timezone("UTC")), axis=1)
    #StanfordTimestamps.set_index('Stanford_timestamp', inplace = True)
    
    tol = pd.Timedelta('1 minute')
    df = pd.merge_asof(left=df.sort_values('Operator_Timestamp'),right=StanfordTimestamps.sort_values('Stanford_timestamp'), right_on='Stanford_timestamp',left_on='Operator_Timestamp',direction='nearest',tolerance=tol)     
    df.loc[df['Stanford_timestamp'].isnull(),'Stanford_timestamp'] = df["Operator_Timestamp"]
    

    return df

def loadMeterData_Bridger(DataPath):

    ## BRIDGER QUADRATHERM + CORIOLIS DATA ## 
    
    # Load time series data from Nanodac:
    # (1) Select data from correct channel depending on which meter was used
    # (2) Delete additional channels
    nano_1_path = os.path.join(DataPath, 'nano_21113_1_exp.csv')
    Quad_data_1 = pd.read_csv(nano_1_path, skiprows=1, usecols=[0,1,2,3,4],names=['datetime_UTC','channel_1','channel_2','channel_3','channel_4'], parse_dates=True)
    Quad_data_1['datetime_UTC'] = pd.to_datetime(Quad_data_1['datetime_UTC'])
    Quad_data_1.set_index('datetime_UTC', inplace = True)
    Quad_data_1['cr_quad_scfh'] = np.nan
    Quad_data_1['PipeSize_inch'] = np.nan
    Quad_data_1['MeterCode'] = np.nan
    Quad_data_1['cr_allmeters_scfh'] = np.nan
    Quad_data_1['Flag_field_recorded'] = False
    #Quad_data_1["cr_allmeters_kgh"] = np.nan
    #Quad_data_1["cr_allmeters_kgh_CH4"] = np.nan
    Quad_data_1['cr_quad_scfh'][(Quad_data_1.index < '2021.11.03 17:33:18')] = Quad_data_1['channel_1'][(Quad_data_1.index < '2021.11.03 17:33:18')]
    Quad_data_1['PipeSize_inch'][(Quad_data_1.index < '2021.11.03 17:33:18')] = 2
    Quad_data_1['MeterCode'][(Quad_data_1.index < '2021.11.03 17:33:18')] = 218645
    Quad_data_1['cr_quad_scfh'][(Quad_data_1.index > '2021.11.03 17:38:37')] = Quad_data_1['channel_2'][(Quad_data_1.index > '2021.11.03 17:38:37')]
    Quad_data_1['PipeSize_inch'][(Quad_data_1.index > '2021.11.03 17:38:37')] = 4
    Quad_data_1['MeterCode'][(Quad_data_1.index > '2021.11.03 17:38:37')] = 308188
    Quad_data_1['cr_quad_scfh'] = pd.to_numeric(Quad_data_1['cr_quad_scfh'],errors = 'coerce')
    #Quad_data_1["cr_allmeters_kgh"] = SCFH2kgh(Quad_data_1['cr_quad_scfh'], T=21.1)
    #Quad_data_1["cr_allmeters_kgh_CH4"] = applyComposition(Quad_data_1["cr_allmeters_kgh"], CH4_frac)
    Quad_data_1["cr_allmeters_scfh"] = Quad_data_1['cr_quad_scfh']
    del Quad_data_1['channel_1'] 
    del Quad_data_1['channel_2']
    del Quad_data_1['channel_3']
    del Quad_data_1['channel_4']
    
    nano_2_path = os.path.join(DataPath, 'nano_21113_2_exp.csv')    
    Quad_data_2 = pd.read_csv(nano_2_path, skiprows=1, usecols=[0,1,2,3,4],names=['datetime_UTC','channel_1','channel_2','channel_3','channel_4'], parse_dates=True)
    Quad_data_2['datetime_UTC'] = pd.to_datetime(Quad_data_2['datetime_UTC'])
    Quad_data_2.set_index('datetime_UTC', inplace = True)
    Quad_data_2['cr_quad_scfh'] = np.nan
    Quad_data_2['PipeSize_inch'] = np.nan
    Quad_data_2['MeterCode'] = np.nan
    Quad_data_2['cr_allmeters_scfh'] = np.nan
    Quad_data_2['Flag_field_recorded'] = False
    #Quad_data_2["cr_allmeters_kgh"] = np.nan
    #Quad_data_2["cr_allmeters_kgh_CH4"] = np.nan
    Quad_data_2['cr_quad_scfh'][(Quad_data_2.index < '2021.11.03 21:32:11')] = Quad_data_2['channel_2'][(Quad_data_2.index < '2021.11.03 21:32:11')]
    Quad_data_2['PipeSize_inch'][(Quad_data_2.index < '2021.11.03 21:32:11')] = 4
    Quad_data_2['MeterCode'][(Quad_data_2.index < '2021.11.03 21:32:11')] = 308188
    Quad_data_2['cr_quad_scfh'] = pd.to_numeric(Quad_data_2['cr_quad_scfh'],errors = 'coerce')
    #Quad_data_2["cr_allmeters_kgh"] = SCFH2kgh(Quad_data_2['cr_quad_scfh'], T=21.1)
    #Quad_data_2["cr_allmeters_kgh_CH4"] = applyComposition(Quad_data_2["cr_allmeters_kgh"], CH4_frac)
    Quad_data_2["cr_allmeters_scfh"] = Quad_data_2['cr_quad_scfh']
    del Quad_data_2['channel_1'] 
    del Quad_data_2['channel_2']
    del Quad_data_2['channel_3']
    del Quad_data_2['channel_4']
    
    nano_3_path = os.path.join(DataPath, 'nano_21114_exp.csv')    
    Quad_data_3 = pd.read_csv(nano_3_path, skiprows=1, usecols=[0,1,2,3,4],names=['datetime_UTC','channel_1','channel_2','channel_3','channel_4'], parse_dates=True)
    Quad_data_3['datetime_UTC'] = pd.to_datetime(Quad_data_3['datetime_UTC'])
    Quad_data_3.set_index('datetime_UTC', inplace = True)
    Quad_data_3['cr_quad_scfh'] = np.nan
    Quad_data_3['PipeSize_inch'] = np.nan
    Quad_data_3['MeterCode'] = np.nan
    Quad_data_3['cr_allmeters_scfh'] = np.nan
    Quad_data_3['cr_Coriolis_gps'] = np.nan
    Quad_data_3['Flag_field_recorded'] = False
    #Quad_data_3['cr_allmeters_kgh'] = np.nan
    #Quad_data_3['cr_allmeters_kgh_CH4'] = np.nan
    Quad_data_3['cr_quad_scfh'][(Quad_data_3.index < '2021.11.04 19:27:38')] = Quad_data_3['channel_2'][(Quad_data_3.index < '2021.11.04 19:27:38')]
    Quad_data_3['cr_quad_scfh'] = pd.to_numeric(Quad_data_3['cr_quad_scfh'][(Quad_data_3.index < '2021.11.04 19:27:38')],errors = 'coerce')
    #Quad_data_3['cr_allmeters_kgh'][(Quad_data_3.index < '2021.11.04 19:27:38')] = SCFH2kgh(Quad_data_3['cr_quad_scfh'][(Quad_data_3.index < '2021.11.04 19:27:38')], T=21.1)
    #Quad_data_3['cr_allmeters_kgh_CH4'][(Quad_data_3.index < '2021.11.04 19:27:38')] = applyComposition(Quad_data_3["cr_allmeters_kgh"][(Quad_data_3.index < '2021.11.04 19:27:38')], CH4_frac)
    Quad_data_3['cr_allmeters_scfh'][(Quad_data_3.index < '2021.11.04 19:27:38')] = Quad_data_3['cr_quad_scfh'][(Quad_data_3.index < '2021.11.04 19:27:38')]
    Quad_data_3['PipeSize_inch'][(Quad_data_3.index < '2021.11.04 19:27:38')] = 4
    Quad_data_3['MeterCode'][(Quad_data_3.index < '2021.11.04 19:27:38')] = 308188          
    Quad_data_3['cr_Coriolis_gps'][(Quad_data_3.index > '2021.11.04 19:28:02') & (Quad_data_3.index < '2021.11.04 20:24:37')] = Quad_data_3['channel_4'][(Quad_data_3.index > '2021.11.04 19:28:02') & (Quad_data_3.index < '2021.11.04 20:24:37')]
    Quad_data_3['cr_Coriolis_gps'] = pd.to_numeric(Quad_data_3['cr_Coriolis_gps'][(Quad_data_3.index > '2021.11.04 19:28:02') & (Quad_data_3.index < '2021.11.04 20:24:37')],errors = 'coerce')
    #Quad_data_3['cr_allmeters_kgh'][(Quad_data_3.index > '2021.11.04 19:28:02') & (Quad_data_3.index < '2021.11.04 20:24:37')] = gps2kgh(Quad_data_3['cr_Coriolis_gps'][(Quad_data_3.index > '2021.11.04 19:28:02') & (Quad_data_3.index < '2021.11.04 20:24:37')])
    #Quad_data_3['cr_allmeters_kgh_CH4'][(Quad_data_3.index > '2021.11.04 19:28:02') & (Quad_data_3.index < '2021.11.04 20:24:37')] = applyComposition(Quad_data_3['cr_allmeters_kgh'][(Quad_data_3.index > '2021.11.04 19:28:02') & (Quad_data_3.index < '2021.11.04 20:24:37')], CH4_frac)
    Quad_data_3['cr_allmeters_scfh'][(Quad_data_3.index > '2021.11.04 19:28:02') & (Quad_data_3.index < '2021.11.04 20:24:37')] = gps2scfh(Quad_data_3['cr_Coriolis_gps'][(Quad_data_3.index > '2021.11.04 19:28:02') & (Quad_data_3.index < '2021.11.04 20:24:37')], T=21.1)
    Quad_data_3['PipeSize_inch'][(Quad_data_3.index > '2021.11.04 19:28:02') & (Quad_data_3.index < '2021.11.04 20:24:37')] = 0.5
    Quad_data_3['MeterCode'][(Quad_data_3.index > '2021.11.04 19:28:02') & (Quad_data_3.index < '2021.11.04 20:24:37')] = 21175085   
    Quad_data_3['PipeSize_inch'][(Quad_data_3.index > '2021.11.04 19:28:02') & (Quad_data_3.index < '2021.11.04 20:24:37')] = 0.5
    Quad_data_3['MeterCode'][(Quad_data_3.index > '2021.11.04 19:28:02') & (Quad_data_3.index < '2021.11.04 20:24:37')] = 21175085   
    Quad_data_3['cr_quad_scfh'][(Quad_data_3.index > '2021.11.04 20:24:45')] = Quad_data_3['channel_2'][(Quad_data_3.index > '2021.11.04 20:24:45')]
    Quad_data_3['cr_quad_scfh'] = pd.to_numeric(Quad_data_3['cr_quad_scfh'][(Quad_data_3.index > '2021.11.04 20:24:45')],errors = 'coerce')
    #Quad_data_3['cr_allmeters_kgh'][(Quad_data_3.index > '2021.11.04 20:24:45')] = SCFH2kgh(Quad_data_3['cr_quad_scfh'][(Quad_data_3.index > '2021.11.04 20:24:45')], T=21.1)
    #Quad_data_3['cr_allmeters_kgh_CH4'][(Quad_data_3.index > '2021.11.04 20:24:45')] = applyComposition(Quad_data_3["cr_allmeters_kgh"][(Quad_data_3.index > '2021.11.04 20:24:45')], CH4_frac)
    Quad_data_3['cr_allmeters_scfh'][(Quad_data_3.index > '2021.11.04 20:24:45')] = Quad_data_3['cr_quad_scfh'][(Quad_data_3.index > '2021.11.04 20:24:45')]
    Quad_data_3['PipeSize_inch'][(Quad_data_3.index > '2021.11.04 20:24:45')] = 4
    Quad_data_3['MeterCode'][(Quad_data_3.index > '2021.11.04 20:24:45')] = 308188    
    del Quad_data_3['channel_1'] 
    del Quad_data_3['channel_2']
    del Quad_data_3['channel_3']
    del Quad_data_3['channel_4']
    #Nanodac appears to have froze for the period 21.11.4 19:28:02 : 19:57:12. Replace data with handwritten notes
    hand_data_21114_path = os.path.join(DataPath, '21114_releasedat_Coriolis.csv')
    hand_data_21114 = pd.read_csv(hand_data_21114_path, skiprows=0, usecols=[0,1],names=['datetime_UTC','cr_Coriolis_gps'], parse_dates=True)
    hand_data_21114['datetime_UTC'] = pd.to_datetime(hand_data_21114['datetime_UTC'])
    hand_data_21114.set_index('datetime_UTC', inplace = True)
    hand_data_21114['cr_Coriolis_gps'] = pd.to_numeric(hand_data_21114['cr_Coriolis_gps'],errors = 'coerce')
    #hand_data_21114['cr_allmeters_kgh'] = gps2kgh(hand_data_21114['cr_Coriolis_gps'])
    #hand_data_21114['cr_allmeters_kgh_CH4'] = applyComposition(hand_data_21114['cr_allmeters_kgh'], CH4_frac) 
    hand_data_21114['cr_allmeters_scfh'] = gps2scfh(hand_data_21114['cr_Coriolis_gps'], T=21.1)
    hand_data_21114['PipeSize_inch'] = 0.5
    hand_data_21114['MeterCode'] = 21175085
    hand_data_21114['Flag_field_recorded'] = True
    

    # Concatenate all time series data
    Quad_data_all = pd.concat([Quad_data_1, Quad_data_2, Quad_data_3])
    
    # Overwrite Nanodac Coriolis data with hand recorded Coriolis data
    Quad_data_all = Quad_data_all.drop(Quad_data_all[(Quad_data_all.index > '2021.11.04 19:28:02') & (Quad_data_all.index < '2021.11.04 19:57:12')].index)
    
    Quad_data_all = pd.concat([Quad_data_all, hand_data_21114])
    
    Quad_date_range_1  = pd.date_range("2021.11.03 16:25:04", periods = 18431, freq = "s")
    Quad_date_range_1 = Quad_date_range_1.to_frame(index = True)
    Quad_date_range_2  = pd.date_range("2021.11.04 16:39:41", periods = 17705, freq = "s")
    Quad_date_range_2 = Quad_date_range_2.to_frame(index = True)
    Quad_date_range = pd.concat([Quad_date_range_1, Quad_date_range_2])
    

    # Perform outer join between date range and Quadratherm data
    quadrathermDF = Quad_date_range.join(Quad_data_all, how='outer')
    time_series = quadrathermDF[0]
    del quadrathermDF[0]
    
    # Back-fill missing data
    quadrathermDF = quadrathermDF.bfill()
    
    # nan data where the Quadratherm isn't being used and where the nanodac isn't being used
    quadrathermDF['cr_Coriolis_gps'][(quadrathermDF.index < '2021.11.04 19:28:02')] = np.NaN
    quadrathermDF['cr_Coriolis_gps'][(quadrathermDF.index > '2021.11.04 20:24:37')] = np.NaN
    quadrathermDF['cr_quad_scfh'][(quadrathermDF.index > '2021.11.04 19:28:02') & (quadrathermDF.index < '2021.11.04 20:24:37')] = np.NaN
    
    # Localize the datetime index
    time_series = time_series.dt.tz_localize(pytz.utc)
    quadrathermDF.index = time_series

    # Add a column for moving average
    #quadrathermDF['cr_allmeters_scfh'] = np.nan
    #quadrathermDF['cr_allmeters_scfh']  = kgh2SCFH(quadrathermDF['cr_allmeters_kgh'], T=21.1)
    quadrathermDF['cr_scfh_mean30'] = quadrathermDF['cr_allmeters_scfh'].rolling(window=30).mean()
    quadrathermDF['cr_scfh_mean60'] = quadrathermDF['cr_allmeters_scfh'].rolling(window=60).mean()
    quadrathermDF['cr_scfh_mean90'] = quadrathermDF['cr_allmeters_scfh'].rolling(window=90).mean()
    #quadrathermDF['cr_kgh_CH4_mean'] = quadrathermDF['cr_allmeters_kgh_CH4'].rolling(window=cr_averageperiod_sec).mean()
    #quadrathermDF['cr_kgh_CH4_std'] = quadrathermDF['cr_allmeters_kgh_CH4'].rolling(window=cr_averageperiod_sec).std()
    #quadrathermDF['cr_coriolis_gps_mean'] = quadrathermDF['instantaneous_Coriolis_gps'].rolling(window=cr_averageperiod_sec).mean()
    #quadrathermDF['cr_coriolis_gps_std'] = quadrathermDF['instantaneous_Coriolis_gps'].rolling(window=cr_averageperiod_sec).std()
    #quadrathermDF['cr_avg_start'] = quadrathermDF.index - datetime.timedelta(seconds = cr_averageperiod_sec)
    #quadrathermDF['cr_avg_end'] = quadrathermDF.index
    
    so_path = os.path.join(DataPath, 'shut_off_stamps.csv')
    shutoff_points = pd.read_csv(so_path, skiprows=0, usecols=[0,1],names=['start_UTC', 'end_UTC'], parse_dates=True)
    shutoff_points['start_UTC'] = pd.to_datetime(shutoff_points['start_UTC'])
    shutoff_points['end_UTC'] = pd.to_datetime(shutoff_points['end_UTC'])
    shutoff_points['start_UTC'] = shutoff_points['start_UTC'].dt.tz_localize(pytz.utc)
    shutoff_points['end_UTC'] = shutoff_points['end_UTC'].dt.tz_localize(pytz.utc)
    
    for i in range(shutoff_points.shape[0]):
        quadrathermDF['cr_allmeters_scfh'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0 
        #quadrathermDF['cr_allmeters_kgh'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        #quadrathermDF['cr_allmeters_kgh_CH4'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        #quadrathermDF['cr_kgh_CH4_mean'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        quadrathermDF['cr_scfh_mean30'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        quadrathermDF['cr_scfh_mean60'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        quadrathermDF['cr_scfh_mean90'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        #quadrathermDF['cr_kgh_CH4_std'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0 
               
    #del quadrathermDF['instantaneous_scfh']  
    #del quadrathermDF['instantaneous_Coriolis_gps']
    
    # Delete all rows with NaT
    quadrathermDF["TMP"] = quadrathermDF.index.values                   # index is a DateTimeIndex
    quadrathermDF = quadrathermDF[quadrathermDF.TMP.notnull()]          # remove all NaT values
    quadrathermDF.drop(["TMP"], axis=1, inplace=True)                   # delete TMP again
    
    quadrathermDF['TestLocation'] = 'AZ'
    
    return quadrathermDF


def loadMeterData_CarbonMapper(DataPath):
 
    ## CARBON MAPPER QUADRATHERM + CORIOLIS DATA ## 

    # Load OCR Data
    OCR_1_path = os.path.join(DataPath, '21730_releasedat.csv')
    Quad_data_1 = pd.read_csv(OCR_1_path, skiprows=1, usecols=[0,1],names=['datetime_local','cr_quad_scfh'], parse_dates=True)
    Quad_data_1['datetime_local'] = pd.to_datetime(Quad_data_1['datetime_local'])
    Quad_data_1['datetime_local'] = Quad_data_1.apply(
        lambda x: pd.NA if pd.isna(x['datetime_local']) else
        x['datetime_local'].replace(tzinfo=pytz.timezone("UTC")), axis=1)
    Quad_data_1.set_index('datetime_local', inplace = True)
    #Quad_data_1["cr_allmeters_kgh"] = np.nan
    #Quad_data_1["cr_allmeters_kgh_CH4"] = np.nan
    Quad_data_1['cr_allmeters_scfh'] = np.nan
    Quad_data_1['PipeSize_inch'] = np.nan
    Quad_data_1['MeterCode'] = np.nan
    Quad_data_1['Flag_field_recorded'] = False
    Quad_data_1['PipeSize_inch'][(Quad_data_1.index < '2021.07.30 17:22:54')] = 2
    Quad_data_1['MeterCode'][(Quad_data_1.index < '2021.07.30 17:22:54')] = 162928
    Quad_data_1['PipeSize_inch'][(Quad_data_1.index >= '2021.07.30 17:22:54')] = 4
    Quad_data_1['MeterCode'][(Quad_data_1.index >= '2021.07.30 17:22:54')] = 218645
    Quad_data_1['cr_quad_scfh'] = pd.to_numeric(Quad_data_1['cr_quad_scfh'],errors = 'coerce')
    #Quad_data_1["cr_allmeters_kgh"] = SCFH2kgh(Quad_data_1['cr_quad_scfh'], T=21.1)
    #Quad_data_1["cr_allmeters_kgh_CH4"] = applyComposition(Quad_data_1["cr_allmeters_kgh"], CH4_frac)
    Quad_data_1['cr_allmeters_scfh'] = Quad_data_1['cr_quad_scfh']
    
    OCR_2_path = os.path.join(DataPath, '2183_releasedat.csv')
    Quad_data_2 = pd.read_csv(OCR_2_path, skiprows=1, usecols=[0,1],names=['datetime_local','cr_quad_scfh'], parse_dates=True)
    Quad_data_2['datetime_local'] = pd.to_datetime(Quad_data_2['datetime_local'])
    Quad_data_2['datetime_local'] = Quad_data_2.apply(
        lambda x: pd.NA if pd.isna(x['datetime_local']) else
        x['datetime_local'].replace(tzinfo=pytz.timezone("UTC")), axis=1)
    Quad_data_2.set_index('datetime_local', inplace = True)
    #Quad_data_2["cr_allmeters_kgh"] = np.nan
    #Quad_data_2["cr_allmeters_kgh_CH4"] = np.nan
    Quad_data_2["cr_allmeters_scfh"] = np.nan
    Quad_data_2['PipeSize_inch'] = np.nan
    Quad_data_2['MeterCode'] = np.nan
    Quad_data_2['Flag_field_recorded'] = False
    Quad_data_2['PipeSize_inch'][(Quad_data_2.index < '2021.08.03 17:35:25')] = 4
    Quad_data_2['MeterCode'][(Quad_data_2.index < '2021.08.03 17:35:25')] = 218645
    Quad_data_2['PipeSize_inch'][(Quad_data_2.index >= '2021.08.03 17:35:25')] = 2
    Quad_data_2['MeterCode'][(Quad_data_2.index >= '2021.08.03 17:35:25')] = 162928  
    Quad_data_2['cr_quad_scfh'] = pd.to_numeric(Quad_data_2['cr_quad_scfh'],errors = 'coerce')
    #Quad_data_2["cr_allmeters_kgh"] = SCFH2kgh(Quad_data_2['cr_quad_scfh'], T=21.1)
    #Quad_data_2["cr_allmeters_kgh_CH4"] = applyComposition(Quad_data_2["cr_allmeters_kgh"], CH4_frac)
    Quad_data_2['cr_allmeters_scfh'] = Quad_data_2['cr_quad_scfh']
    
    # video footage is hard to read on July 31. Use hand recorded data
    # Copied from "GAO" tab of "TX_release_vols_v8.xlsx"
    hand_data_21731_path = os.path.join(DataPath, '21731_releasedat_Quad.csv')
    hand_data_21731 = pd.read_csv(hand_data_21731_path, skiprows=1, usecols=[0,1],names=['datetime_local','cr_quad_scfh'], parse_dates=True)
    hand_data_21731['datetime_local'] = pd.to_datetime(hand_data_21731['datetime_local'])
    hand_data_21731['datetime_local'] = hand_data_21731.apply(
        lambda x: pd.NA if pd.isna(x['datetime_local']) else
        x['datetime_local'].replace(tzinfo=pytz.timezone("US/Central")), axis=1)
    #hand_data_21731["cr_allmeters_kgh"] = np.nan
    #hand_data_21731["cr_allmeters_kgh_CH4"] = np.nan  
    hand_data_21731["cr_allmeters_scfh"] = np.nan 
    hand_data_21731['datetime_local'] = hand_data_21731['datetime_local'].apply(lambda x: x.astimezone(pytz.timezone('UTC')))
    hand_data_21731['PipeSize_inch'] = 4
    hand_data_21731['MeterCode'] = 218645
    hand_data_21731.set_index('datetime_local', inplace = True)
    hand_data_21731['cr_quad_scfh'] = pd.to_numeric(hand_data_21731['cr_quad_scfh'],errors = 'coerce')
    #hand_data_21731["cr_allmeters_kgh"] = SCFH2kgh(hand_data_21731['cr_quad_scfh'], T=21.1)
    #hand_data_21731["cr_allmeters_kgh_CH4"] = applyComposition(hand_data_21731["cr_allmeters_kgh"], CH4_frac)
    hand_data_21731['cr_allmeters_scfh'] = hand_data_21731['cr_quad_scfh']
    hand_data_21731['Flag_field_recorded'] = True

    # Concatenate all time series data
    Quad_data_all = pd.concat([Quad_data_1, Quad_data_2, hand_data_21731])
    
    Quad_date_range_1  = pd.date_range("2021.07.30 15:21:24", periods = 10140, freq = "s")
    Quad_date_range_1 = Quad_date_range_1.tz_localize(pytz.utc)
    Quad_date_range_1 = Quad_date_range_1.to_frame(index = True)
    Quad_date_range_2  = pd.date_range("2021.07.31 15:22:00", periods = 2580, freq = "s")
    Quad_date_range_2 = Quad_date_range_2.tz_localize(pytz.utc)
    Quad_date_range_2 = Quad_date_range_2.to_frame(index = True)
    Quad_date_range_3  = pd.date_range("2021.08.03 15:49:05", periods = 17391, freq = "s")
    Quad_date_range_3 = Quad_date_range_3.tz_localize(pytz.utc)
    Quad_date_range_3 = Quad_date_range_3.to_frame(index = True)
    Quad_date_range = pd.concat([Quad_date_range_1, Quad_date_range_2, Quad_date_range_3])        

    # Perform outer join between date range and Quadratherm data
    quadrathermDF = Quad_date_range.join(Quad_data_all, how='outer')
    time_series = quadrathermDF[0]
    del quadrathermDF[0]
    
    # Back-fill missing data
    quadrathermDF = quadrathermDF.bfill()    

    # Add a column for moving average    
    #quadrathermDF['cr_allmeters_scfh'] = np.nan
    #quadrathermDF['cr_allmeters_scfh']  = kgh2SCFH(quadrathermDF['cr_allmeters_kgh'], T=21.1)
    quadrathermDF['cr_scfh_mean30'] = quadrathermDF['cr_allmeters_scfh'].rolling(window=30).mean()
    quadrathermDF['cr_scfh_mean60'] = quadrathermDF['cr_allmeters_scfh'].rolling(window=60).mean()
    quadrathermDF['cr_scfh_mean90'] = quadrathermDF['cr_allmeters_scfh'].rolling(window=90).mean()
    #quadrathermDF['cr_kgh_CH4_mean'] = quadrathermDF['cr_allmeters_kgh_CH4'].rolling(window=cr_averageperiod_sec).mean()
    #quadrathermDF['cr_kgh_CH4_std'] = quadrathermDF['cr_allmeters_kgh_CH4'].rolling(window=cr_averageperiod_sec).std()

    #quadrathermDF['cr_avg_start'] = quadrathermDF.index - datetime.timedelta(seconds = cr_averageperiod_sec)
    #quadrathermDF['cr_avg_end'] = quadrathermDF.index
    
    so_path = os.path.join(DataPath, 'shut_off_stamps.csv')
    shutoff_points = pd.read_csv(so_path, skiprows=0, usecols=[0,1],names=['start_UTC', 'end_UTC'], parse_dates=True)
    shutoff_points['start_UTC'] = pd.to_datetime(shutoff_points['start_UTC'])
    shutoff_points['end_UTC'] = pd.to_datetime(shutoff_points['end_UTC'])
    shutoff_points['start_UTC'] = shutoff_points['start_UTC'].dt.tz_localize(pytz.utc)
    shutoff_points['end_UTC'] = shutoff_points['end_UTC'].dt.tz_localize(pytz.utc)
    
    for i in range(shutoff_points.shape[0]):
        quadrathermDF['cr_allmeters_scfh'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0 
        #quadrathermDF['cr_allmeters_kgh'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        #quadrathermDF['cr_allmeters_kgh_CH4'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        #quadrathermDF['cr_kgh_CH4_mean'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        quadrathermDF['cr_scfh_mean30'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        quadrathermDF['cr_scfh_mean60'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        quadrathermDF['cr_scfh_mean90'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
       #quadrathermDF['cr_kgh_CH4_std'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0 
    
    quadrathermDF['TestLocation'] = 'TX'
    
    return quadrathermDF    

def loadMeterData_GHGSat(DataPath):

    ## GHGSat QUADRATHERM + CORIOLIS DATA ##     

    # Load OCR data
    OCR_1_path = os.path.join(DataPath, '211018_release_dat.csv')
    Quad_data_1 = pd.read_csv(OCR_1_path, skiprows=1, usecols=[0,1],names=['datetime_local','cr_quad_scfh'], parse_dates=True)
    Quad_data_1['datetime_local'] = pd.to_datetime(Quad_data_1['datetime_local'])
    Quad_data_1['datetime_local'] = Quad_data_1.apply(
        lambda x: pd.NA if pd.isna(x['datetime_local']) else
        x['datetime_local'].replace(tzinfo=pytz.timezone("UTC")), axis=1)
    #Quad_data_1["cr_allmeters_kgh"] = np.nan
    #Quad_data_1["cr_allmeters_kgh_CH4"] = np.nan   
    Quad_data_1['cr_allmeters_scfh'] = np.nan
    Quad_data_1['PipeSize_inch'] = 4
    Quad_data_1['MeterCode'] = 218645
    Quad_data_1['Flag_field_recorded'] = False
    Quad_data_1.set_index('datetime_local', inplace = True)
    Quad_data_1['cr_quad_scfh'] = pd.to_numeric(Quad_data_1['cr_quad_scfh'],errors = 'coerce')
    #Quad_data_1["cr_allmeters_kgh"] = SCFH2kgh(Quad_data_1['cr_quad_scfh'], T=21.1)
    #Quad_data_1["cr_allmeters_kgh_CH4"] = applyComposition(Quad_data_1["cr_allmeters_kgh"], CH4_frac)
    Quad_data_1['cr_allmeters_scfh'] = Quad_data_1['cr_quad_scfh']
    
    OCR_2_path = os.path.join(DataPath, '211019_1_release_dat.csv')
    Quad_data_2 = pd.read_csv(OCR_2_path, skiprows=1, usecols=[0,1],names=['datetime_local','cr_quad_scfh'], parse_dates=True)
    Quad_data_2['datetime_local'] = pd.to_datetime(Quad_data_2['datetime_local'])
    Quad_data_2['datetime_local'] = Quad_data_2.apply(
        lambda x: pd.NA if pd.isna(x['datetime_local']) else
        x['datetime_local'].replace(tzinfo=pytz.timezone("UTC")), axis=1)
    Quad_data_2.set_index('datetime_local', inplace = True)
    #Quad_data_2["cr_allmeters_kgh"] = np.nan
    #Quad_data_2["cr_allmeters_kgh_CH4"] = np.nan   
    Quad_data_2['cr_allmeters_scfh'] = np.nan
    Quad_data_2['PipeSize_inch'] = np.nan
    Quad_data_2['MeterCode'] = np.nan
    Quad_data_2['Flag_field_recorded'] = False
    Quad_data_2['PipeSize_inch'][(Quad_data_2.index < '2021.10.19 18:33:10')] = 8
    Quad_data_2['MeterCode'][(Quad_data_2.index < '2021.10.19 18:33:10')] = 218645
    Quad_data_2['PipeSize_inch'][(Quad_data_2.index >= '2021.10.19 18:33:10')] = 2
    Quad_data_2['MeterCode'][(Quad_data_2.index >= '2021.10.19 18:33:10')] = 218645    
    Quad_data_2['cr_quad_scfh'] = pd.to_numeric(Quad_data_2['cr_quad_scfh'],errors = 'coerce')
    #Quad_data_2["cr_allmeters_kgh"] = SCFH2kgh(Quad_data_2['cr_quad_scfh'], T=21.1)
    #Quad_data_2["cr_allmeters_kgh_CH4"] = applyComposition(Quad_data_2["cr_allmeters_kgh"], CH4_frac)
    Quad_data_2['cr_allmeters_scfh'] = Quad_data_2['cr_quad_scfh']

    hand_3_path = os.path.join(DataPath, '211019_2_release_dat.csv')
    Quad_data_3 = pd.read_csv(hand_3_path, skiprows=1, usecols=[0,1],names=['datetime_local','cr_quad_scfh'], parse_dates=True)
    Quad_data_3['datetime_local'] = pd.to_datetime(Quad_data_3['datetime_local'])
    Quad_data_3['datetime_local'] = Quad_data_3.apply(
        lambda x: pd.NA if pd.isna(x['datetime_local']) else
        x['datetime_local'].replace(tzinfo=pytz.timezone("UTC")), axis=1)
    #Quad_data_3["cr_allmeters_kgh"] = np.nan
    #Quad_data_3["cr_allmeters_kgh_CH4"] = np.nan 
    Quad_data_3['cr_allmeters_scfh'] = np.nan
    Quad_data_3['PipeSize_inch'] = 2
    Quad_data_3['MeterCode'] = 218645
    Quad_data_3.set_index('datetime_local', inplace = True)
    Quad_data_3['cr_quad_scfh'] = pd.to_numeric(Quad_data_3['cr_quad_scfh'],errors = 'coerce')
    #Quad_data_3["cr_allmeters_kgh"] = SCFH2kgh(Quad_data_3['cr_quad_scfh'], T=21.1)
    #Quad_data_3["cr_allmeters_kgh_CH4"] = applyComposition(Quad_data_3["cr_allmeters_kgh"], CH4_frac)  
    Quad_data_3['cr_allmeters_scfh'] = Quad_data_3['cr_quad_scfh']
    Quad_data_3['Flag_field_recorded'] = True

    OCR_4_path = os.path.join(DataPath, '211020_1_release_dat.csv')
    # Units in g/s
    Quad_data_4 = pd.read_csv(OCR_4_path, skiprows=1, usecols=[0,1],names=['datetime_local','cr_Coriolis_gps'], parse_dates=True)  
    Quad_data_4['datetime_local'] = pd.to_datetime(Quad_data_4['datetime_local'])
    Quad_data_4['datetime_local'] = Quad_data_4.apply(
        lambda x: pd.NA if pd.isna(x['datetime_local']) else
        x['datetime_local'].replace(tzinfo=pytz.timezone("UTC")), axis=1)
    #Quad_data_4["cr_allmeters_kgh"] = np.nan
    #Quad_data_4["cr_allmeters_kgh_CH4"] = np.nan 
    Quad_data_4['cr_allmeters_scfh'] = np.nan    
    Quad_data_4['PipeSize_inch'] = 0.5
    Quad_data_4['MeterCode'] = 21175085
    Quad_data_4['Flag_field_recorded'] = False
    Quad_data_4.set_index('datetime_local', inplace = True)
    Quad_data_4['cr_Coriolis_gps'] = pd.to_numeric(Quad_data_4['cr_Coriolis_gps'],errors = 'coerce')
    #Quad_data_4["cr_allmeters_kgh"] = gps2kgh(Quad_data_4['cr_Coriolis_gps'])
    #Quad_data_4["cr_allmeters_kgh_CH4"] = applyComposition(Quad_data_4["cr_allmeters_kgh"], CH4_frac)     
    Quad_data_4['cr_allmeters_scfh'] = gps2scfh(Quad_data_4['cr_Coriolis_gps'], T=21.1)
    
    OCR_5_path = os.path.join(DataPath, '211020_2_release_dat_ocrcorrections.csv')
    Quad_data_5 = pd.read_csv(OCR_5_path, skiprows=1, usecols=[0,1],names=['datetime_local','cr_quad_scfh'], parse_dates=True)
    Quad_data_5['datetime_local'] = pd.to_datetime(Quad_data_5['datetime_local'])
    Quad_data_5['datetime_local'] = Quad_data_5.apply(
        lambda x: pd.NA if pd.isna(x['datetime_local']) else
        x['datetime_local'].replace(tzinfo=pytz.timezone("UTC")), axis=1)
    #Quad_data_5["cr_allmeters_kgh"] = np.nan
    #Quad_data_5["cr_allmeters_kgh_CH4"] = np.nan 
    Quad_data_5['cr_allmeters_scfh'] = np.nan 
    Quad_data_5['PipeSize_inch'] = 2
    Quad_data_5['MeterCode'] = 218645
    Quad_data_5['Flag_field_recorded'] = False
    Quad_data_5.set_index('datetime_local', inplace = True)
    Quad_data_5['cr_quad_scfh'] = pd.to_numeric(Quad_data_5['cr_quad_scfh'],errors = 'coerce')
    #Quad_data_5["cr_allmeters_kgh"] = SCFH2kgh(Quad_data_5['cr_quad_scfh'], T=21.1)
    #Quad_data_5["cr_allmeters_kgh_CH4"] = applyComposition(Quad_data_5["cr_allmeters_kgh"], CH4_frac) 
    Quad_data_5['cr_allmeters_scfh'] = Quad_data_5['cr_quad_scfh']     
   
    OCR_6_path = os.path.join(DataPath, '211021_release_dat.csv')
    Quad_data_6 = pd.read_csv(OCR_6_path, skiprows=1, usecols=[0,1],names=['datetime_local','cr_quad_scfh'], parse_dates=True)
    Quad_data_6['datetime_local'] = pd.to_datetime(Quad_data_6['datetime_local'])
    Quad_data_6['datetime_local'] = Quad_data_6.apply(
        lambda x: pd.NA if pd.isna(x['datetime_local']) else
        x['datetime_local'].replace(tzinfo=pytz.timezone("UTC")), axis=1)
    #Quad_data_6["cr_allmeters_kgh"] = np.nan
    #Quad_data_6["cr_allmeters_kgh_CH4"] = np.nan
    Quad_data_6['cr_allmeters_scfh'] = np.nan 
    Quad_data_6['PipeSize_inch'] = 8
    Quad_data_6['MeterCode'] = 218645
    Quad_data_6['Flag_field_recorded'] = False
    Quad_data_6.set_index('datetime_local', inplace = True)
    Quad_data_6['cr_quad_scfh'] = pd.to_numeric(Quad_data_6['cr_quad_scfh'],errors = 'coerce')
    #Quad_data_6["cr_allmeters_kgh"] = SCFH2kgh(Quad_data_6['cr_quad_scfh'], T=21.1)
    #Quad_data_6["cr_allmeters_kgh_CH4"] = applyComposition(Quad_data_6["cr_allmeters_kgh"], CH4_frac) 
    Quad_data_6['cr_allmeters_scfh'] = Quad_data_6['cr_quad_scfh']

    nano_7_path = os.path.join(DataPath, 'nano_211021_1.csv')
    Quad_data_7 = pd.read_csv(nano_7_path, skiprows=1, usecols=[0,1,2,3,4],names=['datetime_UTC','channel_1','channel_2','channel_3','channel_4'], parse_dates=True)
    Quad_data_7['datetime_UTC'] = pd.to_datetime(Quad_data_7['datetime_UTC'])
    Quad_data_7['datetime_UTC'] = Quad_data_7.apply(
        lambda x: pd.NA if pd.isna(x['datetime_UTC']) else
        x['datetime_UTC'].replace(tzinfo=pytz.timezone("UTC")), axis=1)
    Quad_data_7.set_index('datetime_UTC', inplace = True)
    #Quad_data_7["cr_allmeters_kgh"] = np.nan
    #Quad_data_7["cr_allmeters_kgh_CH4"] = np.nan
    Quad_data_7['cr_allmeters_scfh'] = np.nan
    Quad_data_7['PipeSize_inch'] = 8
    Quad_data_7['MeterCode'] = 218645
    Quad_data_7['Flag_field_recorded'] = False
    Quad_data_7['cr_quad_scfh'] = Quad_data_7['channel_1']
    Quad_data_7['cr_quad_scfh'] = pd.to_numeric(Quad_data_7['cr_quad_scfh'],errors = 'coerce')
    #Quad_data_7["cr_allmeters_kgh"] = SCFH2kgh(Quad_data_7['cr_quad_scfh'], T=21.1)
    #Quad_data_7["cr_allmeters_kgh_CH4"] = applyComposition(Quad_data_7["cr_allmeters_kgh"], CH4_frac) 
    Quad_data_7['cr_allmeters_scfh'] = Quad_data_7['cr_quad_scfh']
    del Quad_data_7['channel_1'] 
    del Quad_data_7['channel_2']
    del Quad_data_7['channel_3']
    del Quad_data_7['channel_4']
    
    nano_8_path = os.path.join(DataPath, 'nano_211021_2.csv')
    Quad_data_8 = pd.read_csv(nano_8_path, skiprows=1, usecols=[0,1,2,3,4],names=['datetime_UTC','channel_1','channel_2','channel_3','channel_4'], parse_dates=True)
    Quad_data_8['datetime_UTC'] = pd.to_datetime(Quad_data_8['datetime_UTC'])
    Quad_data_8['datetime_UTC'] = Quad_data_8.apply(
        lambda x: pd.NA if pd.isna(x['datetime_UTC']) else
        x['datetime_UTC'].replace(tzinfo=pytz.timezone("UTC")), axis=1)
    Quad_data_8.set_index('datetime_UTC', inplace = True)
    #Quad_data_8["cr_allmeters_kgh"] = np.nan
    #Quad_data_8["cr_allmeters_kgh_CH4"] = np.nan
    Quad_data_8['cr_allmeters_scfh'] = np.nan
    Quad_data_8['PipeSize_inch'] = 8
    Quad_data_8['MeterCode'] = 218645
    Quad_data_8['Flag_field_recorded'] = False
    Quad_data_8['cr_quad_scfh'] = Quad_data_8['channel_1']
    Quad_data_8['cr_quad_scfh'] = pd.to_numeric(Quad_data_8['cr_quad_scfh'],errors = 'coerce')
   #Quad_data_8["cr_allmeters_kgh"] = SCFH2kgh(Quad_data_8['cr_quad_scfh'], T=21.1)
   # Quad_data_8["cr_allmeters_kgh_CH4"] = applyComposition(Quad_data_8["cr_allmeters_kgh"], CH4_frac) 
    Quad_data_8['cr_allmeters_scfh'] = Quad_data_8['cr_quad_scfh']
    del Quad_data_8['channel_1'] 
    del Quad_data_8['channel_2']
    del Quad_data_8['channel_3']
    del Quad_data_8['channel_4']
    
    nano_9_path = os.path.join(DataPath, 'nano_211021_3.csv')
    Quad_data_9 = pd.read_csv(nano_9_path, skiprows=1, usecols=[0,1,2,3,4],names=['datetime_UTC','channel_1','channel_2','channel_3','channel_4'], parse_dates=True)
    Quad_data_9['datetime_UTC'] = pd.to_datetime(Quad_data_9['datetime_UTC'])
    Quad_data_9['datetime_UTC'] = Quad_data_9.apply(
        lambda x: pd.NA if pd.isna(x['datetime_UTC']) else
        x['datetime_UTC'].replace(tzinfo=pytz.timezone("UTC")), axis=1)
    Quad_data_9.set_index('datetime_UTC', inplace = True)
    #Quad_data_9["cr_allmeters_kgh"] = np.nan
    #Quad_data_9["cr_allmeters_kgh_CH4"] = np.nan
    Quad_data_9['cr_allmeters_scfh'] = np.nan
    Quad_data_9['PipeSize_inch'] = 8
    Quad_data_9['MeterCode'] = 218645
    Quad_data_9['Flag_field_recorded'] = False
    Quad_data_9['cr_quad_scfh'] = Quad_data_9['channel_1']
    Quad_data_9['cr_quad_scfh'] = pd.to_numeric(Quad_data_9['cr_quad_scfh'],errors = 'coerce')
    #Quad_data_9["cr_allmeters_kgh"] = SCFH2kgh(Quad_data_9['cr_quad_scfh'], T=21.1)
    #Quad_data_9["cr_allmeters_kgh_CH4"] = applyComposition(Quad_data_9["cr_allmeters_kgh"], CH4_frac) 
    Quad_data_9['cr_allmeters_scfh'] = Quad_data_9['cr_quad_scfh']
    del Quad_data_9['channel_1'] 
    del Quad_data_9['channel_2']
    del Quad_data_9['channel_3']
    del Quad_data_9['channel_4']

    nano_10_path = os.path.join(DataPath, 'nano_211021_4.csv')
    Quad_data_10 = pd.read_csv(nano_10_path, skiprows=1, usecols=[0,1,2,3,4],names=['datetime_UTC','channel_1','channel_2','channel_3','channel_4'], parse_dates=True)
    Quad_data_10['datetime_UTC'] = pd.to_datetime(Quad_data_10['datetime_UTC'])
    Quad_data_10['datetime_UTC'] = Quad_data_10.apply(
        lambda x: pd.NA if pd.isna(x['datetime_UTC']) else
        x['datetime_UTC'].replace(tzinfo=pytz.timezone("UTC")), axis=1)
    Quad_data_10.set_index('datetime_UTC', inplace = True)
    #Quad_data_10["cr_allmeters_kgh"] = np.nan
    #Quad_data_10["cr_allmeters_kgh_CH4"] = np.nan
    Quad_data_10['cr_allmeters_scfh'] = np.nan
    Quad_data_10['PipeSize_inch'] = 0.5
    Quad_data_10['MeterCode'] = 21175085
    Quad_data_10['Flag_field_recorded'] = False
    Quad_data_10['cr_Coriolis_gps'] = Quad_data_10['channel_4']
    Quad_data_10['cr_Coriolis_gps'] = pd.to_numeric(Quad_data_10['cr_Coriolis_gps'],errors = 'coerce')
    #Quad_data_10["cr_allmeters_kgh"] = gps2kgh(Quad_data_10['cr_Coriolis_gps'])
    #Quad_data_10["cr_allmeters_kgh_CH4"] = applyComposition(Quad_data_10["cr_allmeters_kgh"], CH4_frac) 
    Quad_data_10['cr_allmeters_scfh'] = gps2scfh(Quad_data_10['cr_Coriolis_gps'], T=21.1)
    del Quad_data_10['channel_1'] 
    del Quad_data_10['channel_2']
    del Quad_data_10['channel_3']
    del Quad_data_10['channel_4']
    
    nano_11_path = os.path.join(DataPath, 'nano_211022.csv')
    Quad_data_11 = pd.read_csv(nano_11_path, skiprows=1, usecols=[0,1,2,3,4],names=['datetime_UTC','channel_1','channel_2','channel_3','channel_4'], parse_dates=True)
    Quad_data_11['datetime_UTC'] = pd.to_datetime(Quad_data_11['datetime_UTC'])
    Quad_data_11['datetime_UTC'] = Quad_data_11.apply(
        lambda x: pd.NA if pd.isna(x['datetime_UTC']) else
        x['datetime_UTC'].replace(tzinfo=pytz.timezone("UTC")), axis=1)
    Quad_data_11.set_index('datetime_UTC', inplace = True)
    #Quad_data_11["cr_allmeters_kgh"] = np.nan
    #Quad_data_11["cr_allmeters_kgh_CH4"] = np.nan
    Quad_data_11['cr_allmeters_scfh'] = np.nan
    Quad_data_11['PipeSize_inch'] = 4
    Quad_data_11['MeterCode'] = 218645
    Quad_data_11['Flag_field_recorded'] = False
    Quad_data_11['cr_quad_scfh'] = Quad_data_11['channel_2']
    Quad_data_11['cr_quad_scfh'] = pd.to_numeric(Quad_data_11['cr_quad_scfh'],errors = 'coerce')
    #Quad_data_11["cr_allmeters_kgh"] = SCFH2kgh(Quad_data_11['cr_quad_scfh'], T=21.1)
    #Quad_data_11["cr_allmeters_kgh_CH4"] = applyComposition(Quad_data_11["cr_allmeters_kgh"], CH4_frac) 
    Quad_data_11['cr_allmeters_scfh'] = Quad_data_11['cr_quad_scfh']
    del Quad_data_11['channel_1'] 
    del Quad_data_11['channel_2']
    del Quad_data_11['channel_3']
    del Quad_data_11['channel_4']

    # Concatenate all time series data
    Quad_data_all = pd.concat([Quad_data_1, Quad_data_2, Quad_data_3, Quad_data_4, Quad_data_5, Quad_data_6, Quad_data_7, Quad_data_8, Quad_data_9, Quad_data_10, Quad_data_11])

    idx_18  = pd.date_range("2021.10.18 16:55:46", periods = 9199, freq = "s")
    idx_18 = idx_18.tz_localize(pytz.utc)
    idx_18 = idx_18.to_frame(index = True)
    idx_19  = pd.date_range("2021.10.19 17:40:00", periods = 10439, freq = "s")
    idx_19 = idx_19.tz_localize(pytz.utc)
    idx_19 = idx_19.to_frame(index = True)
    idx_20  = pd.date_range("2021.10.20 17:43:28", periods = 11648, freq = "s")
    idx_20 = idx_20.tz_localize(pytz.utc)
    idx_20 = idx_20.to_frame(index = True)    
    idx_21  = pd.date_range("2021.10.21 17:43:41", periods = 12394, freq = "s")
    idx_21 = idx_21.tz_localize(pytz.utc)
    idx_21= idx_21.to_frame(index = True)
    idx_22  = pd.date_range("2021.10.22 16:38:09", periods = 16214, freq = "s")
    idx_22 = idx_22.tz_localize(pytz.utc)
    idx_22 = idx_22.to_frame(index = True)
    Quad_date_range = pd.concat([idx_18, idx_19, idx_20, idx_21, idx_22])

    # Perform outer join between date range and Quadratherm data
    quadrathermDF = Quad_date_range.join(Quad_data_all, how='outer')
    time_series = quadrathermDF[0]
    del quadrathermDF[0]
    
    # Back-fill missing data
    quadrathermDF = quadrathermDF.bfill()

    # nan data where the Quadratherm isn't being used and where the nanodac isn't being used
    quadrathermDF['cr_Coriolis_gps'][(quadrathermDF.index < '2021.10.20 17:43:38')] = np.NaN
    quadrathermDF['cr_Coriolis_gps'][(quadrathermDF.index > '2021.10.20 19:47:52') & (quadrathermDF.index < '2021.10.21 19:52:09')] = np.NaN
    quadrathermDF['cr_Coriolis_gps'][(quadrathermDF.index > '2021.10.21 20:48:03')] = np.NaN

    # Add a column for moving average  
    #quadrathermDF['cr_allmeters_scfh'] = np.nan
    #quadrathermDF['cr_allmeters_scfh']  = kgh2SCFH(quadrathermDF['cr_allmeters_kgh'], T=21.1)
    quadrathermDF['cr_scfh_mean30'] = quadrathermDF['cr_allmeters_scfh'].rolling(window=30).mean()
    quadrathermDF['cr_scfh_mean60'] = quadrathermDF['cr_allmeters_scfh'].rolling(window=60).mean()
    quadrathermDF['cr_scfh_mean90'] = quadrathermDF['cr_allmeters_scfh'].rolling(window=90).mean()
    #quadrathermDF['cr_kgh_CH4_mean'] = quadrathermDF['cr_allmeters_kgh_CH4'].rolling(window=cr_averageperiod_sec).mean()
    #quadrathermDF['cr_kgh_CH4_std'] = quadrathermDF['cr_allmeters_kgh_CH4'].rolling(window=cr_averageperiod_sec).std()
    
    #quadrathermDF['cr_avg_start'] = quadrathermDF.index - datetime.timedelta(seconds = cr_averageperiod_sec)
    #quadrathermDF['cr_avg_end'] = quadrathermDF.index
    
    so_path = os.path.join(DataPath, 'shut_off_stamps.csv')
    shutoff_points = pd.read_csv(so_path, skiprows=0, usecols=[0,1],names=['start_UTC', 'end_UTC'], parse_dates=True)
    shutoff_points['start_UTC'] = pd.to_datetime(shutoff_points['start_UTC'])
    shutoff_points['end_UTC'] = pd.to_datetime(shutoff_points['end_UTC'])
    shutoff_points['start_UTC'] = shutoff_points['start_UTC'].dt.tz_localize(pytz.utc)
    shutoff_points['end_UTC'] = shutoff_points['end_UTC'].dt.tz_localize(pytz.utc)
    
    for i in range(shutoff_points.shape[0]):  
        quadrathermDF['cr_allmeters_scfh'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0 
        #quadrathermDF['cr_allmeters_kgh'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        #quadrathermDF['cr_allmeters_kgh_CH4'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        #quadrathermDF['cr_kgh_CH4_mean'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        quadrathermDF['cr_scfh_mean30'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        quadrathermDF['cr_scfh_mean60'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        quadrathermDF['cr_scfh_mean90'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        #quadrathermDF['cr_kgh_CH4_std'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0 
          
        
    quadrathermDF['TestLocation'] = 'AZ'
    
    return quadrathermDF    


def combineAnemometer_Bridger(sonic_path):
    
    # Process data for November 3
    # Location = Ehrenberg
    # Timezone = MST
    
    # Sonic data is in Palo Alto time
    localtz = pytz.timezone("US/Pacific")
    date_string = '21.11.3'
    path_lookup = sonic_path + date_string + '\\'
    #path_export = path_compiled + date_string
    cols = [1,2,6]
    AZ_day = 19
    offset = 0.8438*AZ_day + 54.865
    offset = int(round(offset))

    sonic_date_range_1  = pd.date_range("2021.11.03 16:59:08", periods = 17115, freq = "s")
    sonic_date_range_1 = sonic_date_range_1.to_frame(index = True)
    sonic_date_range_1  = sonic_date_range_1.tz_localize(pytz.utc)

    sonicDF_temp1 = processAnemometer(path_lookup, localtz, cols, offset) 
    sonicDF_temp1 = sonicDF_temp1.set_index('time')
    
    # Perform outer join between date range and Quadratherm data
    #sonicDF_temp1 = sonic_date_range_1.join(sonicDF_temp1, how='outer')
    sonicDF_temp1 = sonic_date_range_1.merge(sonicDF_temp1, how='outer', left_index=True, right_index=True)
    sonicDF_temp1 = sonicDF_temp1.iloc[: , 2:]
    
     # Back-fill missing data
    sonicDF_temp1 = sonicDF_temp1.bfill()
    
    # Process data for November 4
    # Location = Ehrenberg
    # Timezone = MST
    
    # Sonic data is in Palo Alto time
    localtz = pytz.timezone("US/Pacific")
    date_string = '21.11.4'
    path_lookup = sonic_path + date_string + '\\'
    #path_export = path_compiled + date_string
    cols = [1,2,6]
    AZ_day = 20
    offset = 0.8438*AZ_day + 54.865
    offset = int(round(offset))

    sonic_date_range_2  = pd.date_range("2021.11.04 15:57:57", periods = 20367, freq = "s")
    sonic_date_range_2 = sonic_date_range_2.to_frame(index = True)
    sonic_date_range_2  = sonic_date_range_2.tz_localize(pytz.utc)
    
    sonicDF_temp2 = processAnemometer(path_lookup, localtz, cols, offset)     
    sonicDF_temp2 = sonicDF_temp2.set_index('time')
    
    # Perform outer join between date range and Quadratherm data
    #sonicDF_temp1 = sonic_date_range_1.join(sonicDF_temp1, how='outer')
    sonicDF_temp2 = sonic_date_range_2.merge(sonicDF_temp2, how='outer', left_index=True, right_index=True)
    sonicDF_temp2 = sonicDF_temp2.iloc[: , 2:]
    
     # Back-fill missing data
    sonicDF_temp2 = sonicDF_temp2.bfill()
    
    sonicDF = pd.concat([sonicDF_temp1, sonicDF_temp2])



    return sonicDF

def combineAnemometer_CarbonMapper(sonic_path):
    
    # Process data for July 30
    # Location = Midland
    # Timezone = US/Central
    
    # Sonic data is in Palo Alto time
    localtz = pytz.timezone("US/Pacific")
    date_string = '21.7.30'
    path_lookup = sonic_path + date_string + '\\'
    #path_export = path_compiled + date_string
    cols = [1,2,6]
    offset = 3
    offset = int(round(offset))

    sonic_date_range_1  = pd.date_range("2021.07.30 14:37:48", periods = 14422, freq = "s")
    sonic_date_range_1 = sonic_date_range_1.to_frame(index = True)
    sonic_date_range_1  = sonic_date_range_1.tz_localize(pytz.utc)

    sonicDF_temp1 = processAnemometer(path_lookup, localtz, cols, offset) 
    sonicDF_temp1 = sonicDF_temp1.set_index('time')
    
    # Perform outer join between date range and Quadratherm data
    #sonicDF_temp1 = sonic_date_range_1.join(sonicDF_temp1, how='outer')
    sonicDF_temp1 = sonic_date_range_1.merge(sonicDF_temp1, how='outer', left_index=True, right_index=True)
    sonicDF_temp1 = sonicDF_temp1.iloc[: , 2:]
    
     # Back-fill missing data
    sonicDF_temp1 = sonicDF_temp1.bfill()
    
    # Process data for July 31
    # Location = Midland
    # Timezone = US/Central
    
    # Sonic data is in Palo Alto time
    localtz = pytz.timezone("US/Pacific")
    date_string = '21.7.31'
    path_lookup = sonic_path + date_string + '\\'
    #path_export = path_compiled + date_string
    cols = [1,2,7]
    offset = 3
    offset = int(round(offset))

    sonic_date_range_2  = pd.date_range("2021.07.31 14:20:58", periods = 19695, freq = "s")
    sonic_date_range_2 = sonic_date_range_2.to_frame(index = True)
    sonic_date_range_2  = sonic_date_range_2.tz_localize(pytz.utc)
    
    sonicDF_temp2 = processAnemometer(path_lookup, localtz, cols, offset)     
    sonicDF_temp2 = sonicDF_temp2.set_index('time')
    
    # Perform outer join between date range and Quadratherm data
    #sonicDF_temp1 = sonic_date_range_1.join(sonicDF_temp1, how='outer')
    sonicDF_temp2 = sonic_date_range_2.merge(sonicDF_temp2, how='outer', left_index=True, right_index=True)
    sonicDF_temp2 = sonicDF_temp2.iloc[: , 2:]
    
     # Back-fill missing data
    sonicDF_temp2 = sonicDF_temp2.bfill()

    # Process data for August 3
    # Location = Midland
    # Timezone = US/Central
    
    # Sonic data is in Palo Alto time
    localtz = pytz.timezone("US/Pacific")
    date_string = '21.8.3'
    path_lookup = sonic_path + date_string + '\\'
    #path_export = path_compiled + date_string
    cols = [1,2,7]
    offset = 3
    offset = int(round(offset))

    sonic_date_range_3  = pd.date_range("2021.08.03 14:21:30", periods = 24448, freq = "s")
    sonic_date_range_3 = sonic_date_range_3.to_frame(index = True)
    sonic_date_range_3  = sonic_date_range_3.tz_localize(pytz.utc)
    
    sonicDF_temp3 = processAnemometer(path_lookup, localtz, cols, offset)     
    sonicDF_temp3 = sonicDF_temp3.set_index('time')
    
    # Perform outer join between date range and Quadratherm data
    #sonicDF_temp1 = sonic_date_range_1.join(sonicDF_temp1, how='outer')
    sonicDF_temp3 = sonic_date_range_3.merge(sonicDF_temp3, how='outer', left_index=True, right_index=True)
    sonicDF_temp3 = sonicDF_temp3.iloc[: , 2:]
    
     # Back-fill missing data
    sonicDF_temp3 = sonicDF_temp3.bfill()
    
    sonicDF = pd.concat([sonicDF_temp1, sonicDF_temp2, sonicDF_temp3])


    return sonicDF

def combineAnemometer_GHGSat(sonic_path):
    
    # Process data for October 18
    # Location = Ehrenberg
    # Timezone = MST
    
    # Sonic data is in Palo Alto time
    localtz = pytz.timezone("US/Pacific")
    date_string = '21.10.18'
    path_lookup = sonic_path + date_string + '\\'
    #path_export = path_compiled + date_string
    cols = [1,2,6]
    AZ_day = 3
    offset = 0.8438*AZ_day + 54.865
    offset = int(round(offset))

    sonic_date_range_1  = pd.date_range("2021.10.18 17:04:49", periods = 9830, freq = "s")
    sonic_date_range_1 = sonic_date_range_1.to_frame(index = True)
    sonic_date_range_1  = sonic_date_range_1.tz_localize(pytz.utc)

    sonicDF_temp1 = processAnemometer(path_lookup, localtz, cols, offset) 
    sonicDF_temp1 = sonicDF_temp1.set_index('time')
    
    # Perform outer join between date range and Quadratherm data
    #sonicDF_temp1 = sonic_date_range_1.join(sonicDF_temp1, how='outer')
    sonicDF_temp1 = sonic_date_range_1.merge(sonicDF_temp1, how='outer', left_index=True, right_index=True)
    sonicDF_temp1 = sonicDF_temp1.iloc[: , 2:]
    
     # Back-fill missing data
    sonicDF_temp1 = sonicDF_temp1.bfill()
    
    # Process data for October 19
    # Location = Ehrenberg
    # Timezone = MST
    
    # Sonic data is in Palo Alto time
    localtz = pytz.timezone("US/Pacific")
    date_string = '21.10.19'
    path_lookup = sonic_path + date_string + '\\'
    #path_export = path_compiled + date_string
    cols = [1,2,6]
    AZ_day = 4
    offset = 0.8438*AZ_day + 54.865
    offset = int(round(offset))

    sonic_date_range_2  = pd.date_range("2021.10.19 17:20:38", periods = 16952, freq = "s")
    sonic_date_range_2 = sonic_date_range_2.to_frame(index = True)
    sonic_date_range_2  = sonic_date_range_2.tz_localize(pytz.utc)
    
    sonicDF_temp2 = processAnemometer(path_lookup, localtz, cols, offset)     
    sonicDF_temp2 = sonicDF_temp2.set_index('time')
    
    # Perform outer join between date range and Quadratherm data
    #sonicDF_temp1 = sonic_date_range_1.join(sonicDF_temp1, how='outer')
    sonicDF_temp2 = sonic_date_range_2.merge(sonicDF_temp2, how='outer', left_index=True, right_index=True)
    sonicDF_temp2 = sonicDF_temp2.iloc[: , 2:]
    
     # Back-fill missing data
    sonicDF_temp2 = sonicDF_temp2.bfill()

    # Process data for October 20
    # Location = Ehrenberg
    # Timezone = MST
    
    # Sonic data is in Palo Alto time
    localtz = pytz.timezone("US/Pacific")
    date_string = '21.10.20'
    path_lookup = sonic_path + date_string + '\\'
    #path_export = path_compiled + date_string
    cols = [1,2,6]
    AZ_day = 5
    offset = 0.8438*AZ_day + 54.865
    offset = int(round(offset))

    sonic_date_range_3  = pd.date_range("2021.10.20 17:50:15", periods = 17614, freq = "s")
    sonic_date_range_3 = sonic_date_range_3.to_frame(index = True)
    sonic_date_range_3  = sonic_date_range_3.tz_localize(pytz.utc)
    
    sonicDF_temp3 = processAnemometer(path_lookup, localtz, cols, offset)     
    sonicDF_temp3 = sonicDF_temp3.set_index('time')
    
    # Perform outer join between date range and Quadratherm data
    #sonicDF_temp1 = sonic_date_range_1.join(sonicDF_temp1, how='outer')
    sonicDF_temp3 = sonic_date_range_3.merge(sonicDF_temp3, how='outer', left_index=True, right_index=True)
    sonicDF_temp3 = sonicDF_temp3.iloc[: , 2:]
    
     # Back-fill missing data
    sonicDF_temp3 = sonicDF_temp3.bfill()

    # Process data for October 21
    # Location = Ehrenberg
    # Timezone = MST
    
    # Sonic data is in Palo Alto time
    localtz = pytz.timezone("US/Pacific")
    date_string = '21.10.21'
    path_lookup = sonic_path + date_string + '\\'
    #path_export = path_compiled + date_string
    cols = [1,2,6]
    AZ_day = 6
    offset = 0.8438*AZ_day + 54.865
    offset = int(round(offset))

    sonic_date_range_4  = pd.date_range("2021.10.21 16:27:33", periods = 16917, freq = "s")
    sonic_date_range_4 = sonic_date_range_4.to_frame(index = True)
    sonic_date_range_4  = sonic_date_range_4.tz_localize(pytz.utc)
    
    sonicDF_temp4 = processAnemometer(path_lookup, localtz, cols, offset)     
    sonicDF_temp4 = sonicDF_temp4.set_index('time')
    
    # Perform outer join between date range and Quadratherm data
    #sonicDF_temp1 = sonic_date_range_1.join(sonicDF_temp1, how='outer')
    sonicDF_temp4 = sonic_date_range_4.merge(sonicDF_temp4, how='outer', left_index=True, right_index=True)
    sonicDF_temp4 = sonicDF_temp4.iloc[: , 2:]
    
     # Back-fill missing data
    sonicDF_temp4 = sonicDF_temp4.bfill()

    # Process data for October 22
    # Location = Ehrenberg
    # Timezone = MST
    
    # Sonic data is in Palo Alto time
    localtz = pytz.timezone("US/Pacific")
    date_string = '21.10.22'
    path_lookup = sonic_path + date_string + '\\'
    #path_export = path_compiled + date_string
    cols = [1,2,6]
    AZ_day = 7
    offset = 0.8438*AZ_day + 54.865
    offset = int(round(offset))

    sonic_date_range_5  = pd.date_range("2021.10.22 16:39:00", periods = 17036, freq = "s")
    sonic_date_range_5 = sonic_date_range_5.to_frame(index = True)
    sonic_date_range_5  = sonic_date_range_5.tz_localize(pytz.utc)
    
    sonicDF_temp5 = processAnemometer(path_lookup, localtz, cols, offset)     
    sonicDF_temp5 = sonicDF_temp5.set_index('time')
    
    # Perform outer join between date range and Quadratherm data
    #sonicDF_temp1 = sonic_date_range_1.join(sonicDF_temp1, how='outer')
    sonicDF_temp5 = sonic_date_range_5.merge(sonicDF_temp5, how='outer', left_index=True, right_index=True)
    sonicDF_temp5 = sonicDF_temp5.iloc[: , 2:]
    
     # Back-fill missing data
    sonicDF_temp5 = sonicDF_temp5.bfill()
    
    sonicDF = pd.concat([sonicDF_temp1, sonicDF_temp2, sonicDF_temp3, sonicDF_temp4, sonicDF_temp5])



    return sonicDF

def processAnemometer(path_lookup, localtz, cols, offset):

    #os.chdir(path_lookup)    
    #os.listdir()[2:]

    data = pd.DataFrame()
    for     file in os.listdir(path_lookup)[0:]:
          file_data = pd.read_csv(os.path.join(path_lookup, file),skiprows=4,
                        usecols=cols,names=['Direction','Speed_MPS','time'], index_col='time', parse_dates=True)
          file_data = file_data.dropna()
          data = data.append(file_data)
    
    data = data.reset_index()

    df = data.copy()
    df['Speed_MPS'] = df['Speed_MPS'].astype(float)

    # Set timezone
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].dt.tz_localize(localtz)
    df = df.sort_values("time").reset_index(drop=True)    
    df['time'] = df['time'].apply(lambda x: x.astimezone(pytz.utc))

    # Apply time offset    
    df['time'] = df['time'] - datetime.timedelta(seconds = offset)

    # Calculate moving average of wind speed
    df['Speed_Moving_MPS'] = df['Speed_MPS'].rolling(window =300).mean()
    
    
    return df
    