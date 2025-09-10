import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import calendar
from scipy.interpolate import interp1d


def cyclic_interp(values, query_value):
    """
    Interpolate monthly value using periodic boundary conditions.
    """
    x = np.arange(0.5, 13, 1)
    y = np.append(values, values[0])
    wrapped = 0.5 + (query_value - 0.5) % 12.0
    return interp1d(x, y, kind='linear')(wrapped)


class Corso_emissions:
    def __init__(self, csv_path):

        def read_csv(csv_file, drop_last_col=False):
            # Parse CSV with Pandas.
            df = pd.read_csv(f'{csv_path}/{csv_file}', sep=';', header=0, index_col=0)
            return df.iloc[:,:-1] if drop_last_col else df

        # Read hour/week/month and vertical profiles.
        self.df_hour  = read_csv('corso_ps_hourly_profiles_v2.0.csv', True)
        self.df_week  = read_csv('corso_ps_weekly_profiles_v2.0.csv', True)
        self.df_month = read_csv('corso_ps_monthly_profiles_v2.0.csv', True)
        self.df_vert  = read_csv('corso_ps_vertical_profiles_v2.0.csv', True)

        # Read emissions.
        self.df_emiss  = read_csv('corso_ps_catalogue_v2.0.csv')


    def filter_emissions(self, lon_min, lon_max, lat_min, lat_max):
        """
        Filter emission dataframe (`self.df_emiss`) on lat/lon bounds.
        """
        lat = self.df_emiss.latitude
        lon = self.df_emiss.longitude

        mask = (lon > lon_min) & (lon < lon_max) & (lat > lat_min) & (lat < lat_max)

        self.df_emiss = self.df_emiss[mask]


    def get_emission(self, id, specie, date, interpolate_month=True):
        """
        Get emission for requested location and specie, with scaling from time profiles applied.
        Resulting value is in `kg(specie) / s`,
        """
        # Only full hours are supported for now to avoid interpolations.
        if not (date.minute == 0 and date.second == 0 and date.microsecond == 0):
            raise Exception('Only full hours are supported for now!')

        # Month, weekday, and hour as integers.
        # 0 = Januari and 0 = Monday, 0 = 00:00 UTC.
        index_month = date.month - 1
        index_day = date.weekday()
        index_hour = date.hour

        # Days per month.
        days_per_month = calendar.monthrange(date.year, date.month)[1]

        # Filter Dataframe on emission ID.
        dfs = self.df_emiss.loc[id]

        # Get time profile IDs.
        id_month = dfs['ID_MonthFact']
        id_week  = dfs['ID_WeekFact']
        id_hour  = dfs['ID_HourFact']

        # Get time profiles as Numpy arrays with floats.
        def get_prof(df, id):
            return df[df.index == id].values.astype(float).flatten()

        prof_month = get_prof(self.df_month, id_month)
        prof_week  = get_prof(self.df_week, id_week)
        prof_hour  = get_prof(self.df_hour, id_hour)

        # Total emission in kg per year.
        emiss_y = dfs[f'{specie}_kty'] * 10**6

        # Emission in kg / month.
        if interpolate_month:
            month_frac = index_month + date.day / days_per_month
            month_scale_fac = cyclic_interp(prof_month, month_frac)
            emiss_m = emiss_y * month_scale_fac / 12.
        else:
            emiss_m = emiss_y * prof_month[index_month] / 12.

        # Emission in kg / day.
        emiss_d = emiss_m / days_per_month * prof_week[index_day]

        # Emission in kg / hour.
        emiss_h = emiss_d * prof_hour[index_hour] / 24.

        # Check: OKAY!
        print(emiss_h, emiss_y / 365.25 / 24)

        # Return emission in kg / second.
        return emiss_h / 3600


    def get_profile(self, id):
        """
        Get emission profile for requested emission ID.
        """
        # Filter Dataframe on emission ID.
        dfs = self.df_emiss.loc[id]

        # Get time profile IDs.
        id_prof = dfs['ID_VertProf']

        # Get time profiles as Numpy arrays with floats.
        def get_prof(df, id):
            return df[df.index == id].values.astype(float).flatten()

        # Emission heights.
        z = np.arange(50, 1500, 100)
        z = np.append(z, 1500)

        return z, get_prof(self.df_vert, id_prof)


if __name__ == '__main__':
    """
    Just for testing..
    """

    e = Corso_emissions('/home/scratch1/bart/emissions/corso')
    e.filter_emissions(4, 5, 50, 51)

    for index, row in e.df_emiss.iterrows():
        ze, pe = e.get_profile(index)
        em = e.get_emission(index, 'co2', datetime(2022, 1, 1, 12))