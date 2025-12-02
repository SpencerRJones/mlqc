
satellite = 'GCOMW1'
sensor = 'AMSR2'

feature_descriptions = [
    '6V', '6H', '7V', '7H', '10V', '10H', 
    '19V', '19H', '24V', '24H', '37V', '37H'
]

channel_descriptions = feature_descriptions

nedt = [
    0.34, 0.34, 0.43, 0.43, 0.70, 0.70, 0.70, 0.70, 
    0.60, 0.60, 0.70, 0.70
]

nfeatures = len(feature_descriptions)
nchannels = nfeatures
nscangroups = 1
