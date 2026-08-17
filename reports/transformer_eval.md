# Transformer baseline evaluation (dev split)

Model: distilbert-base-cased
Entity-level micro F1: 0.513

```
               precision    recall  f1-score   support

  corporation      0.185     0.147     0.164        34
creative-work      0.156     0.067     0.093       105
        group      0.143     0.179     0.159        39
     location      0.581     0.581     0.581        74
       person      0.768     0.619     0.686       470
      product      0.418     0.202     0.272       114

    micro avg      0.598     0.450     0.513       836
    macro avg      0.375     0.299     0.326       836
 weighted avg      0.574     0.450     0.500       836

```