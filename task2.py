"""
This script exemplifies the usage of the LanguageIdentifier model, using just one line of code to perform model training
This script outputs a file containing all the predictions for the file described in the test_data_path variable
"""
from langid.language_identifier import LanguageIdentifier

model = LanguageIdentifier()

language_data_files = ['langid-data/task2/data.pt-br', 'langid-data/task2/data.pt-pt']
language_data_codes = ['pt-br', 'pt-pt']

test_data_path = 'langid/langid-variants.test'
output_data_path = 'langid/langid-variants.predictions'

# Single line of code performs model training for the listed input files
model.train(language_data_files, language_data_codes, num_samples_per_file=20000, n_gram_range=(1,3))

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
