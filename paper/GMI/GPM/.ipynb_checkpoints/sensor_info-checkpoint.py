
satellite = 'GPM'
sensor = 'GMI'

feature_descriptions = [
    '10V', '10H', '19V', '19H', '24V', '37V', '37H', 
    '89V', '89H', '166V', '166H', '183+-3V', '183+-7V'
]

channel_descriptions = feature_descriptions

nedt = [
    0.77, 0.78, 0.63, 0.60, 0.51, 0.41, 0.42,
    0.32, 0.31, 0.7, 0.65, 0.56, 0.47 
]

nfeatures = len(feature_descriptions)
nchannels = nfeatures
nscangroups = 2
