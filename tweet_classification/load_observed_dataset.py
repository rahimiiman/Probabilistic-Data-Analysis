from datasets import load_dataset

dataset = load_dataset("hate_speech_offensive")

# dataset in highest level is a dictionary with keys:
print("dataset in highest level is a dictionary with keys:")
print(list(dataset.keys()))

#  dataset only has train key , so you need to split the dataset into train and test

print("number of samples in the dataset:", len(dataset["train"]))

print ("each sample is of type: ", type(dataset["train"][0]))

print("each sample is a dictionary with keys: ", dataset["train"].features.keys())


#count is the total number of votes for each sample (tweet)
# hate_speech_count is the number of votes for hate speech
# offensive_language_count is the number of votes for offensive language
# neither_count is the number of votes for neither
# class is the label for the sample (0 for hate speech, 1 for offensive language, 2 for neither)
# tweet is the text of the sample (tweet)

# lets check two sample data as instance
print("sample data 1: ", dataset["train"][0])
print(dataset["train"][0]['tweet'])
print(32*'-')
print("sample data 2: ", dataset["train"][1])
print(dataset["train"][1]['tweet'])