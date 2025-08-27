#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#%%
df1 = pd.read_csv('Trail1_extracted_features_acceleration_m1ai1-1.csv')
df2 = pd.read_csv('Trail2_extracted_features_acceleration_m1ai1.csv')
df3 = pd.read_csv('Trail3_extracted_features_acceleration_m2ai0.csv')
# %%
combined_data = pd.concat([df1, df2, df3])
combined_data.to_csv('combined_trails_features.csv', index=False)
# %%
columns_to_remove = ['start_time', 'axle', 'cluster', 'tsne_1', 'tsne_2']
cleaned_up_data = combined_data.drop(columns=columns_to_remove)
cleaned_up_data["event"] = cleaned_up_data["event"].map(lambda x: 0 if x == "normal" else 1)
cleaned_up_data.to_csv('cleaned_up_data_trails_features.csv', index=False)
# %%
for_normalization = cleaned_up_data.drop(columns=['event'])
event_column = cleaned_up_data['event']  

min_val = for_normalization.min(axis=0)
max_val = for_normalization.max(axis=0)

normalized_w_out_event = (for_normalization - min_val) / (max_val - min_val)
normalized = normalized_w_out_event
normalized['event'] = event_column
normalized.to_csv('normalized.csv', index=False)
# %%
X = normalized.drop(columns=['event'])
y = normalized['event']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123456, stratify=y)
X_train.to_csv("features_train.csv", index=False)
X_test.to_csv("features_test.csv", index=False)
y_train.to_csv("event_train.csv", index=False)
y_test.to_csv("event_test.csv", index=False)
#%%
svm_clf = SVC()

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)
cross_val_res = cross_val_score(svm_clf, X_train, y_train, cv=kf, scoring='accuracy')

svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

fold_ids = np.arange(1, len(cross_val_res) + 1)

plt.figure(figsize=(7, 4))
plt.plot(fold_ids, cross_val_res, marker='o')
plt.xticks(fold_ids)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('K-fold Accuracy, 5-fold')
plt.grid(True)
plt.tight_layout()
plt.savefig('IAI-5fold-accuracy.png')
plt.close()
#%%
print("K-fold:")
print("accuracies")
for i, score in enumerate(cross_val_res, start=1):
    print(f"  Fold {i}: {score:.3f}")
print(f"Mean accuracy: {cross_val_res.mean():.4f},  std accuracy: {cross_val_res.std():.4f}")

print("\n80/20:")
print(f"accuracy: {test_accuracy:.4f}")

print("\ncompared:")
print(f"K-Fold mean accuracy: {cross_val_res.mean():.3f}")
print(f"80/20 accuracy: {test_accuracy:.3f}")


# %%
correlations = pd.concat([X_train, y_train.rename("event")], axis=1).corr()["event"].drop("event")

abs_corr = correlations.abs().sort_values(ascending=False)

top_k = 7
top_features = abs_corr.head(top_k).index.tolist()

print(f"features by Perason correlation:")
print(top_features)

X_train_pearson = X_train[top_features]
X_test_pearson  = X_test[top_features]

svm_clf.fit(X_train_pearson, y_train)
y_pred_pearson = svm_clf.predict(X_test_pearson)

accuracy_pearson = accuracy_score(y_test, y_pred_pearson)
print(f"\n Pearson correlation accuracy: {accuracy_pearson:.4f}")
# %%
model = LinearSVC()

rfe = RFE(estimator=model, n_features_to_select=top_k)
rfe.fit(X_train, y_train)

rfe_features = X_train.columns[rfe.support_].tolist()
print(f"features selected by RFE:")
print(rfe_features)

X_train_rfe = X_train[rfe_features]
X_test_rfe  = X_test[rfe_features]

svm_clf.fit(X_train_rfe, y_train)
y_pred_rfe = svm_clf.predict(X_test_rfe)

accuracy_rfe = accuracy_score(y_test, y_pred_rfe)
print(f"\nRFE accuracy: {accuracy_rfe:.4f}")
# %%
lasso = LogisticRegression(penalty="l1", solver="liblinear")
lasso.fit(X_train, y_train)

abs_coef = np.abs(lasso.coef_).ravel()
lasso_order = pd.Series(abs_coef, index=X_train.columns).sort_values(ascending=False)

lasso_features = lasso_order.head(top_k).index.tolist()
print(f"features selected by LASSO:")
print(lasso_features)

X_train_lasso = X_train[lasso_features]
X_test_lasso  = X_test[lasso_features]

svm_clf.fit(X_train_lasso, y_train)
y_pred_lasso = svm_clf.predict(X_test_lasso)

accuracy_lasso = accuracy_score(y_test, y_pred_lasso)
print(f"\nLASSO accuracy: {accuracy_lasso:.4f}")
#%%
decisiontree = DecisionTreeClassifier(random_state=123456)
decisiontree.fit(X_train, y_train)

decisiontree_importance = pd.Series(decisiontree.feature_importances_, index=X_train.columns)
decisiontree_order = decisiontree_importance.sort_values(ascending=False)

k_grid = [5, 7, 9, 11, 13]
k_grid = [k for k in k_grid if k <= X_train.shape[1]]

best_k, best_cv = None, -1.0
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)

for k in k_grid:
    feats_k = decisiontree_order.head(k).index.tolist()
    scores = cross_val_score(svm_clf, X_train[feats_k], y_train, cv=cv, scoring='accuracy')
    print(f"features = {k}, CV accuracy mean: {scores.mean():.3f}")
    if scores.mean() > best_cv:
        best_cv, best_k = scores.mean(), k

best_features = decisiontree_order.head(best_k).index.tolist()
print("features:", list(best_features))
print(f"\nbest number of features = {best_k}, mean accuracy: {best_cv:.3f}")

X_train_dt = X_train[best_features]
X_test_dt  = X_test[best_features]

svm_clf.fit(X_train_dt, y_train)
y_pred_dt = svm_clf.predict(X_test_dt)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"\nDecision tree accuracy: {accuracy_dt:.4f}")
# %%
