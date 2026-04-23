
sentiment classifier : in this Repo we have trained a model to classify sentences as normal or hate/offensive
we have used the "hate_speech_offensive" dataset from Hugging Face datasets library
the model is trained using BERT base uncased model and fine-tuned on the dataset for sequence classification task
the training code is in "train.py" file and the testing code is in "test.py" file
the model and tokenizer are saved in "sentiment_classifier" directory after training

if you dont know the datasert and want to explore it you can run "load_observed_dataset.py" file it will load the dataset and print some information about it and show some sample data 

if you want to test the model and see the result you can run "test.py"  or "test2.py" file
the "test.py" file will allow you to enter sentences interactively and classify them one by one 
while the "test2.py" file will read sentences from "input.txt" file and classify them and save the results in "output.txt" file in the format:
sentence --> label

the model is trained and saved in following link : you can download it and paste it in tweet_classifier folder
https://mega.nz/folder/FRxilRRb#B6zNbi2ez_uoMhhgt1DunQ


if you want to train the model you can run "train.py" file it will load the dataset, preprocess it, split it into train and test sets, fine-tune the BERT model on the training set and evaluate it on the test set and save the model and tokenizer in "sentiment_classifier" directory

