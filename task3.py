"""
This script performs training and detection of code switching
"""
from code_switching.code_switching import CodeSwitching

# Instantiate the model
model = CodeSwitching()

# Performs model training
# COMMENT THESE TWO LINES IF YOU ALREADY HAVE A TRAINED MODEL
model.train('code_switching/data/train_data.tsv')
model.save_model('model.pkl')

# UNCOMMEBT THIS LINE IF YOU ALREADY HAVE A TRAINED MODEL
#model.load_model('model.pkl')

# Performs prediction of an input string
result = model.predict('Good Morning and welcome to the jungle')

print(result)
