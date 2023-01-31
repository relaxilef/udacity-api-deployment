# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model aims to predict the categorical income of a person based on recorded census data using a logistic regression model.

## Intended Use
This model should be used for academic research only and is not suited for a production environment.

## Training Data
The model has been trained on roughly 26,000 examples from the Census dataset.

## Evaluation Data
The model has been testes on roughly 6,500 examples from the Census dataset.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
The model's performance was evaluated to be

| **Precision** | **Recall** | **Fbeta** |
| :-----------: | :--------: | :-------: |
|     0.64     |   0.32    |   0.43    |

For those interested in detailed slice performances, please cf. `src/data/slice_output.txt`.

## Ethical Considerations
As this academic research example is not intended for the usage in production, no ethical considerations need to be raised at this point.

## Caveats and Recommendations
I recommend taking more time for evaluating better preprocessing steps and to take everything with a grain of salt. ;)