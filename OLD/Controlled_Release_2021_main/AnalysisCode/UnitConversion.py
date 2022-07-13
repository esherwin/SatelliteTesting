import pandas as pd


def convertUnits(df, CH4_frac):
    """Convert units in each data field as desired"""

    # apply composition correction.
    df['cr_mole_frac'] = CH4_frac
    df = applyComposition(df, CH4_frac)

    # Quadratherm reference conditions 1 SCFH (21.1C, 1 atm)
    df['cr_kgh_CH4_mean'] = df.apply(lambda x: pd.NA if pd.isna(x['cr_scfh_CH4_mean']) else SCFH2kgh(x['cr_scfh_CH4_mean'], T=21.1), axis=1)
    df['cr_kgh_CH4_std'] = df.apply(lambda x: pd.NA if pd.isna(x['cr_scfh_CH4_std']) else SCFH2kgh(x['cr_scfh_CH4_std'], T=21.1), axis=1)
    df['cr_kgh_CH4_mean'][(df['Match Time'] > '2021.11.04 19:28:02') & (df['Match Time'] < '2021.11.04 20:24:37')] = df.apply(
        lambda x: pd.NA if pd.isna(x['cr_coriolis_gps_mean']) else gps2kgh(x['cr_coriolis_gps_mean']), axis=1)
    df['cr_kgh_CH4_std'][(df['Match Time'] > '2021.11.04 19:28:02') & (df['Match Time'] < '2021.11.04 20:24:37')] = df.apply(
        lambda x: pd.NA if pd.isna(x['cr_coriolis_gps_std']) else gps2kgh(x['cr_coriolis_gps_std']), axis=1)

    # set flow meter
    df['FlowController'] = 'Quadratherm'
    df['FlowController'][(df['Match Time'] > '2021.11.04 19:28:02') & (df['Match Time'] < '2021.11.04 20:24:37')] = 'Coriolis'

    # Bridger reference conditions (20C, 1 atm)
    df['b_kgh'] = df.apply(lambda x: pd.NA if pd.isna(x['Emission Rate (SCFH)']) else SCFH2kgh(x['Emission Rate (SCFH)'], T=20), axis=1)

    # Windspeed used by bridger
    df['b_DetectionWindSpeed_ms'] = df.apply(
        lambda x: pd.NA if pd.isna(x['Detection Wind Speed (mph)']) else mph2ms(x['Detection Wind Speed (mph)']), axis=1)
    df['b_WindSpeedAt10m_ms'] = df.apply(lambda x: mph2ms(x['Wind Speed 10m (mph)']), axis=1)

    # # Wind Speed measured at release site
    # df['ff_WindSpeed_ms_mean'] = df.apply(
    #     lambda x: pd.NA if pd.isna(x['ff_windspeedMPH_mean']) else mph2ms(x['ff_windspeedMPH_mean']), axis=1)

    return df

def SCFH2kgh(SCFH, T, P=101.325):
    """Convert SCFH to kg/h given "standard" conditions T and P
    :param SCFH = flow rate in standard cubic feet per hour to convert
    :param T = Standard temperature (C)
    :param P = Standard Pressure (kPa) """
    M = 16.04  # molecular weight of methane (g/mol)
    v = 22.4   # 22.4 L (@ 0 C, 1 atm)/1 mol
    kv = 28.3168  # L/ft^3
    km = 1000  # g/kg
    Tratio = 273.15/(T+273.15)
    Pratio = P/101.325
    kgh = SCFH*M/km*kv/v*Tratio*Pratio  # [ft3/h][g/mol]*[kg/g]*[L/ft3]*[mol/L]*[1]*[1]
    return kgh

def kgh2SCFH(kgh, T, P=101.325):
    """Convert kg/h to SCFH given "standard" conditions T and P
    :param SCFH = flow rate in standard cubic feet per hour to convert
    :param T = Standard temperature (C)
    :param P = Standard Pressure (kPa) """
    M = 16.04  # molecular weight of methane (g/mol)
    v = 22.4   # 22.4 L (@ 0 C, 1 atm)/1 mol
    kv = 28.3168  # L/ft^3
    km = 1000  # g/kg
    Tratio = 273.15/(T+273.15)
    Pratio = P/101.325
    SCFH = kgh * (1/(M/km*kv/v*Tratio*Pratio))
    return SCFH

def mph2ms(mph):
    """Convert mph to m/s"""
    return mph*0.44704

def gps2kgh(gps):
    """Convert mph to m/s"""
    return gps*(3600/1000)

def gps2scfh(gps, T, P=101.325):
    """Convert gps to scfh"""
    kgh = gps*(3600/1000)
    
    M = 16.04  # molecular weight of methane (g/mol)
    v = 22.4   # 22.4 L (@ 0 C, 1 atm)/1 mol
    kv = 28.3168  # L/ft^3
    km = 1000  # g/kg
    Tratio = 273.15/(T+273.15)
    Pratio = P/101.325
    SCFH = kgh * (1/(M/km*kv/v*Tratio*Pratio))
    
    return SCFH


def applyComposition(KGH, CH4_frac):
    """Apply composition correction to Alicat readings."""
    
    KGH_CH4 = KGH * CH4_frac
    return KGH_CH4



