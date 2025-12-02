
satellite = 'F17'
sensor = 'SSMIS'

feature_descriptions = [
    '19V', '19H', '22V', '37V', '37H', '92Va', '92Ha', '92Vb', '92Hb',
    '150Ha', '150Hb', '183+-1Ha', '183+-1Hb', '183+-3Ha', '183+-3Hb', '183+-7Ha', '183+-7Hb'
]

channel_descriptions = feature_descriptions

nedt = [
    0.7, 0.7, 0.7, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 
    0.88, 0.88, 1.25, 1.25, 1.0, 1.0, 1.2, 1.2 
]

nfeatures = len(feature_descriptions)
nchannels = nfeatures
nscangroups = 4
