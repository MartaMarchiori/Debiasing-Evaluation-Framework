# Investigating Debiasing Effects on Classification and Explainability

During each stage of a dataset creation and development process, harmful biases can be accidentally introduced, leading to models that perpetuates marginalization and discrimination of minorities, as the role of the data used during the training is critical. 
We propose an evaluation framework that investigates the impact on classification and explainability of bias mitigation preprocessing techniques used to assess data imbalances concerning minorities' representativeness and mitigate the skewed distribution discovered.
Our evaluation focuses on assessing fairness, explainability and performance metrics.
A key dimension of the framework concerns comparing classifiers trained on datasets with the complete set of features and blind datasets, i.e., the sensitive attribute is masked and not accessible. 
We analyze the behavior of local model-agnostic explainers on the original and mitigated datasets to examine whether the proxy models learned by the explainability techniques to mimic the black-boxes disproportionately rely on sensitive attributes, demonstrating biases rooted in the explainers. 
We conduct several experiments about known biased datasets to demonstrate our proposal’s novelty and effectiveness for evaluation and bias detection purposes. 

> Marta Marchiori Manerba and Riccardo Guidotti. "Investigating Debiasing Effects on Classification and Explainability". AIES, 2022.
