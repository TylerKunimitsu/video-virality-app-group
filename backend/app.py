from state import data

import runpy

data['title'] = data['title'].fillna('')
data['description'] = data['description'].fillna('')
data['tags'] = data['tags'].fillna('')

#runpy.run_path('preprocessing/text/sentiment_tokens.py')
#runpy.run_path('preprocessing/text/semantics.py')
#runpy.run_path('preprocessing/text/title.py')
runpy.run_path('preprocessing/text/tags.py')