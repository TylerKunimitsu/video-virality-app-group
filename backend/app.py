from state import data

import runpy

data['description'] = data['description'].fillna('')

runpy.run_path('preprocessing/title-descriptions/sentiment_tokens.py')