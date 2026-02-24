from state import data

import runpy

data['title'] = data['title'].fillna('')
data['description'] = data['description'].fillna('')

#runpy.run_path('preprocessing/title-descriptions/sentiment_tokens.py')
#runpy.run_path('preprocessing/title-descriptions/semantics.py')
#runpy.run_path('preprocessing/title-descriptions/title.py')