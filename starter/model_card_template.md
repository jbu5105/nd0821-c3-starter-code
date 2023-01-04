# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
For this classification problem of `RandomForestClassifier` is used from `scikit-learn`.

## Intended Use
The classification model is trained to classify employee's salary into salaries greater or smaller and equal to 50k.


## Training and Evaluation Data
To train this model `Census Bureau` dataset is used for training and evaluation.

The train, validation split policiy is 80-20.

## Metrics
Precision, recall and f1-score are used to evaluate the model. TResults:

Precision: 0.7266666666666667
Recall: 0.6329032258064516
F1-score: 0.6765517241379311

## Ethical Considerations
There might be some bias in the dataset with respect to sex, race and native-country. Beware of this when using this model on a real application.

## Caveats and Recommendations
There is some sampling bias in this Dataset as it represent a sample from the USA which does not apply to other countries. Beware of this.

We recommend to perform hyperparameter tuning to obtain better metric scores.
