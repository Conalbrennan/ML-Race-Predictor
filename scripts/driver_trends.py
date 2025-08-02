import fastf1
import pandas as pd

def get_driver_stats_and_trends(year, target_round, num_past_races=4):
    """
    For a given year and round, returns a DataFrame with:
    - average qualifying time
    - average finish position
    - DNF rate
    For each driver, based on the last `num_past_races` races before `target_round`.
    """
    qualifying_times = []
    race_results = []

    rounds_to_include = range(max(1, target_round - num_past_races), target_round)

    for rnd in rounds_to_include:
        try:
            # Load qualifying session
            quali = fastf1.get_session(year, rnd, 'Q')
            quali.load()
            quali_laps = quali.laps
            fastest_qualis = quali_laps.groupby('Driver')['LapTime'].min().dropna().dt.total_seconds().reset_index()
            fastest_qualis['Round'] = rnd
            qualifying_times.append(fastest_qualis)

            # Load race session
            race = fastf1.get_session(year, rnd, 'R')
            race.load()
            race_results_df = race.results[['Abbreviation', 'Position', 'Status']].copy()
            race_results_df.rename(columns={
                'Abbreviation': 'Driver',
                'Position': 'FinishPosition'
            }, inplace=True)
            race_results_df['DNF'] = race_results_df['Status'].apply(lambda s: 0 if s == 'Finished' else 1)
            race_results_df['Round'] = rnd
            race_results.append(race_results_df)
        except Exception as e:
            print(f"⚠️ Failed to load round {rnd}: {e}")
            continue

    if not qualifying_times or not race_results:
        return pd.DataFrame(columns=[
            'Driver', 'avg_qualifying_time', 'avg_finish_position', 'dnf_rate'
        ])

    # Combine all rounds
    quali_all = pd.concat(qualifying_times)
    race_all = pd.concat(race_results)

    # Aggregate per driver
    quali_agg = quali_all.groupby('Driver')['LapTime'].mean().reset_index()
    quali_agg.rename(columns={'LapTime': 'avg_qualifying_time'}, inplace=True)

    race_agg = race_all.groupby('Driver').agg({
        'FinishPosition': 'mean',
        'DNF': 'mean'
    }).reset_index()
    race_agg.rename(columns={
        'FinishPosition': 'avg_finish_position',
        'DNF': 'dnf_rate'
    }, inplace=True)

    # Merge into a single DataFrame
    stats_df = pd.merge(quali_agg, race_agg, on='Driver', how='outer')

    return stats_df
