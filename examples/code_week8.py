import finalproject

file_name = "Location1.py"
df, basic_stats = finalproject.load_observations_data(file_name)


variable_name = 'windspeed_100m'
site_index = 1
starting_time = '2017-05-01 00:00:00'
ending_time = '2017-05-01 23:59:59'
fig, ax = finalproject.plot_timeseries(variable_name, site_index, starting_time, ending_time)