# Procedural Fairness and Its Relationship with Distributive Fairness in Machine Learning

This is the code for the paper "Procedural Fairness and Its Relationship with Distributive Fairness in Machine Learning", in which we propose an in-process method to achieve procedural fairness, and analyze the relationship between procedural fairness and distributive fairness in ML.

### Personal Use Only. No Commercial Use.

Part of the code that is based on the "Procedural Fairness in Machine Learning": ([https://github.com/bottydim/adversarial_explanations](https://github.com/oddwang/GPF-FAE)).
## Running experiments

Achieve procedural fairness

```
python procedural_fairness_model.py
```

RQ1 What are the influences of inherent dataset bias and the ML model's procedural fairness on its distributive fairness?

```
python unbiased_data_procedural_fair_model.py
python biased_data_procedural_fair_model.py
python unbiased_data_procedural_unfair_model.py
python biased_data_procedural_unfair_model.py
```

RQ2 What are the differences between optimizing procedural fairness metrics and distributive fairness metrics in ML?

```
python procedural_fairness_model.py
python optimize_distributive_fairness.py
```

## Dependencies

We require the following dependencies:
- aif360==0.5.0
- dill==0.3.7
- Keras==2.3.1
- lime==0.2.0.1
- matplotlib==3.5.3
- numpy==1.21.6
- pandas==1.1.5
- scikit_learn==1.0.2
- scipy==1.7.3
- seaborn==0.13.2
- shap==0.41.0
- tensorflow==1.14.0
- torch==1.12.1
