# Fake News Classifier

## Introduction
Exercise to practice Machine Learning and NLP focused on creating a Fake News classifier.

This exercise was a collaboration between [Desirée Maestri](https://github.com/d-maestri/), [Jan Dirk](https://github.com/JanDirkvandeBijl) and [Jonathan Sada](https://github.com/jonathansada).

## Project Overview
The objective of this project is to build a classifier that is able to distinguish between the Fake (0) and Real(1) headline news.

Once a model is trained it must be used to predict the data in the file `dataset/testing_data.csv`.

## Dataset
- `training_data.csv`: Dataset provided to train and test the model.
- `testing_data.csv`: Dataset to provide the answer of the exercise. It contains all labels set as 2. The purpose of the exercise is to replace them by 0 (fake) or 1 (real) according to the model predictions.

## Result and Conclusions 
The project is splitted in different main files:
- `fake_news_classifier.ipynb`: Contains the step by step of the Machine Learning model creation and comparission (requires and environment with `requirements-notebook.txt` installed).
- `fine_tuning.py` and `fine-tuning-bert.py`: Train a fine-tuned model based on bert and takes predictions (requires and environment with `requirements-fine_tuning.txt` installed)
- `FakeNewsClassifier.pdf`: contains the presentation, technologies used, model comparission, fine-tuned models details and results.

The folder `output` contains csv with some of the results of the different secctions of the exercise and the final answer of the exercise:
- `improve_pred_results.csv`: Contains the metrcis for training and test for all the combinations of classifiers and vectorizers tested.
- `n-grams_results.csv`: Contains the metrics for training and test for the different combinations of ngrams tested in the 2 best models identified on the previous file.
- `logisticregression_model_validation.csv`: Contains the predictions done by the model selected in previous steps in iterations where words with higher impact in the preictions were removed. This was done to validate the model was not dependant on the words with high coefficients.
- `pre_trained_models.csv`: Contains the predictions taken on different fake news classifiers pre-trained models from Hugging Face.
- `alternative_data_cleanning.csv`: Contais prediction metrics done to compare how different cleaning aproaches affect the model selected.
- `testing_data.csv`: Contains the resuling predictions of our model (the provided dataset with the labels updated)

The folder `result` contails the predictions done by the bert fine-tunned model

## Credits
Pre-Trained models used:
- [bert-tiny-finetuned-fake-news-detection](https://huggingface.co/mrm8488/bert-tiny-finetuned-fake-news-detection) by [Manuel Romero](https://huggingface.co/mrm8488)
- [Fake-News-Bert-Detect](https://huggingface.co/jy46604790/Fake-News-Bert-Detect) by [Jiayi Fu](https://huggingface.co/jy46604790)
- [distilbert_fake_news](https://huggingface.co/yasmine-11/distilbert_fake_news) by [yasmine-11](https://huggingface.co/yasmine-11)

## License 
This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/). \
![CC BY 4.0](https://i.creativecommons.org/l/by/4.0/88x31.png) 
