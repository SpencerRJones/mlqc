
satellite = 'NOAA20'
sensor = 'ATMS'

feature_descriptions = [
    '24QV', '31QV', '88QV', '166QH', '183+-7QH', '183+-5QH', '183+-3QH', '183+-2QH', '183+-1QH'   
]

channel_descriptions = feature_descriptions

nedt = [
    0.9, 0.9, 0.5, 0.6, 0.8, 0.8, 0.8, 0.8, 0.9
]

nfeatures = len(feature_descriptions)
nchannels = nfeatures
nscangroups = 4
