"""
This script exemplifies the usage of the LanguageIdentifier model, including all the steps to train the model and
an example of prediction using an input file.
This script outputs a file containing all the predictions for the file described in the test_data_path variable
"""
from langid.language_identifier import LanguageIdentifier

model = LanguageIdentifier()

language_data_files = ['langid-data/task1/data.en', 'langid-data/task1/data.es', 'langid-data/task1/data.pt']
language_data_codes = ['en', 'es', 'pt']

test_data_path = 'langid/langid.test'
output_data_path = 'langid/langid.predictions'

# Load all the data to the model's class
data = model.load_data(language_data_files, language_data_codes, num_samples_per_file=20000)

# Performs sentence tokenization and celanup
data = model.tokenize(data)
data = model.clean_sentences(data)

# Train the model with the prepared training data
model.fit(data.clean, data.code_num)

# Opens the test data file and performs language identification
out_codes = []
with open(test_data_path, encoding='utf8') as f:
    lines = f.readlines()
    for line in lines:
        result = model.predict([line])
        for res in result:
            out_codes.append(language_data_codes[res])
with open(output_data_path, 'w') as of:
    for out_code in out_codes:
        of.write(out_code)
        of.write('\n')
    of.close()