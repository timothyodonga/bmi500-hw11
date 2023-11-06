#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#%%
sex_splits = [0.3, 0.4, 0.5, 0.6, 0.7]
print(sex_splits)

# %%
# Sex Statistics - SBP and DBP distributions
# Men
sbp_male_mean = 133.0
sbp_male_std = 18.6
sbp_male_variance = sbp_male_std**2

dbp_male_mean = 78.8
dbp_male_std = 12.6
dbp_male_variance = dbp_male_std**2

corr_coef_sbp_dbp_male = 0.45
covariance_sbp_dbp_male = corr_coef_sbp_dbp_male * sbp_male_std * dbp_male_std

mean_male = [sbp_male_mean, dbp_male_mean]
cov_male = [[sbp_male_variance, covariance_sbp_dbp_male], [covariance_sbp_dbp_male, dbp_male_variance]]

# Female 
sbp_female_mean = 125.8
sbp_female_std = 19.0
sbp_female_variance = sbp_male_std**2

dbp_female_mean = 74.8
dbp_female_std = 12.4
dbp_female_variance = dbp_female_std**2

corr_coef_sbp_dbp_female = 0.5
covariance_sbp_dbp_female = corr_coef_sbp_dbp_female * sbp_female_std * dbp_female_std

mean_female = [sbp_female_mean, dbp_female_mean]
cov_female = [[sbp_female_variance, covariance_sbp_dbp_female], [covariance_sbp_dbp_female, dbp_female_variance]]

#%%
N = int(1e5)
accuracy_  = []
f1_score_ = []
fpr_ = []
tpr_ = []
thresholds_ = []
roc_auc_ = []


for sex_split in sex_splits:
    # # Add a random seed
    np.random.seed(0)
    print(f"Sex split: {sex_split}")

    M = int(sex_split*N)
    F = int((1-sex_split)*N)

    # Preparing the dataset
    sbp_male, dbp_male = np.random.multivariate_normal(mean_male, cov_male, M).T
    male_labels = np.ones(M)
    sbp_female, dbp_female = np.random.multivariate_normal(mean_female, cov_female, F).T
    female_labels = np.zeros(F)

    df = pd.DataFrame(data={'sbp': list(sbp_male) + list(sbp_female), 'dbp':list(dbp_male) + list(dbp_female), 'label': list(male_labels) + list(female_labels)})
    #  Preprocess the data
    # Shuffle the rows
    df = df.sample(frac = 1)

    X = df[df.columns.difference(['label'])]
    y = df['label']

  

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)


    # Create a model to train
    # model = LogisticRegression(penalty='elasticnet',solver='saga',random_state=0, max_iter=200, l1_ratio=0.5).fit(X_train, y_train)
    model = LogisticRegression(class_weight='balanced')
    # model = SVC(kernel='rbf', random_state=4)
    model.fit(X_train, y_train)


    # Evaluate the model
    from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc
    from sklearn import metrics

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    f1 = f1_score(y_pred, y_test, average='macro')
    accuracy_.append(accuracy)
    f1_score_.append(f1)


    y_pred_ = model.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_)
    auc_value = auc(fpr, tpr)

    fpr_.append(fpr)
    tpr_.append(tpr)

    thresholds_.append(thresholds)
    roc_auc_.append(auc_value)



results = pd.DataFrame(data = {
    'sex_split': sex_splits,
    'Accuracy': accuracy_,
    'f1 score': f1_score_,
    'auc': roc_auc_,
})

results.to_csv('results.csv', index=False)
plt.rcParams['figure.figsize'] = [12,8]
plt.rcParams.update({'font.size':10})


for i in range(len(results)):
    plt.figure()
    plt.plot(fpr_[i], tpr_[i],linestyle='-', label=f'{sex_splits[i]*100:.0f}% male , AUC={roc_auc_[i]:.4f}')
    plt.title(f'ROC curve for Logistic regression classifier for {sex_splits[i]*100:.0f}% male prevalence')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'roc_curve_{i}.png')
