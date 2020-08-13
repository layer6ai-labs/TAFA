# Why_I_like_it Improvement
For quick run/debug:
```bash
python main.py --model [model]
```
To tune hyperparameters:
```bash
python tune_parameters_new.py -y config/test.yml
```
where test.yml looks something like this:
```bash
parameters:
    model: rnn_autorec
    lam: [1, 10, 100, 000]
    rank: [50, 100, 200, 500]
    rec_batch_size: [100]
    rec_epoch: [10]
    iteration: [10]
    lang_learning_rate: [0.0001]
    rec_learning_rate: [0.0001]
    lang_feature_batch_size: [32]
    predict_loss_positive_only: [0]
    fix_encoder: [0, 1]
    topK: [[5, 10, 15, 20, 50]]
    criteria: [NDCG]
    metric: [[R-Precision, NDCG, Precision, Recall]]
```
Check utility/modelnames_new.py for all model names
