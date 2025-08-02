import fastf1
import pandas as pd

def get_austria_race_data(year=2024):
    session = fastf1.get_session(year, 'Austria', 'R')
    session.load()

    laps = session.laps
    results = session.results

    drivers = results['Abbreviation'].values
    data = []

    for driver in drivers:
        driver_laps = laps.pick_driver(driver)

        if driver_laps.empty:
            continue

        # Extract driver features
        fastest_lap = driver_laps['LapTime'].min().total_seconds()
        compound = driver_laps['Compound'].mode()[0] if not driver_laps['Compound'].mode().empty else 'UNKNOWN'
        dnf = int(results[results.Abbreviation == driver]['Status'].values[0] != 'Finished')
        position = int(results[results.Abbreviation == driver]['Position'])

        data.append({
            'Driver': driver,
            'FastestLap': fastest_lap,
            'Compound': compound,
            'DNF': dnf,
            'RacePosition': position
        })

    return pd.DataFrame(data)
