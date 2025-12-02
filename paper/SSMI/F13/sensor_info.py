
satellite = 'F13'
sensor = 'SSMI'

feature_descriptions = [
    '19V', '19H', '22V', '37V', '37H', '85Va', '85Ha', '85Vb', '85Hb'
]

channel_descriptions = feature_descriptions

nedt = [
    0.6, 0.6, 0.6, 0.6, 0.6, 1.1, 1.1, 1.1, 1.1
]

nfeatures = len(feature_descriptions)
nchannels = nfeatures
nscangroups = 2
