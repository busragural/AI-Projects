from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# Define parameter grid for Gradient Boosting
param_grid_gbm = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

def trainGBM(X_train, X_test, y_train, y_test):

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Model Parameters:", model.get_params())

    y_pred = model.predict(X_test)

    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    return metrics

def trainGBMGrid(X_train, X_test, y_train, y_test):

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    gbm = GradientBoostingClassifier(random_state=42)

    grid_search_gbm = GridSearchCV(estimator=gbm, param_grid=param_grid_gbm, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

    grid_search_gbm.fit(X_train, y_train)

    best_model_gbm = grid_search_gbm.best_estimator_
    best_params_gbm = grid_search_gbm.best_params_

    y_pred = best_model_gbm.predict(X_test)

    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    print("Best Params:", best_params_gbm)

    return metrics

def trainGBMSMOTE(X_train, X_test, y_train, y_test):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    smote = SMOTE(random_state=1)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = GradientBoostingClassifier(learning_rate=0.1, max_depth=7, min_samples_leaf=4, min_samples_split=10, n_estimators=100)
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_test)
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    print("Model Parameters:", model.get_params())
    return metrics

def trainGBMSMOTEGrid(X_train, X_test, y_train, y_test):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    smote = SMOTE(random_state=1)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    gbm = GradientBoostingClassifier(random_state=42)
    grid_search_gbm = GridSearchCV(estimator=gbm, param_grid=param_grid_gbm, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

    grid_search_gbm.fit(X_resampled, y_resampled)

    best_model_gbm = grid_search_gbm.best_estimator_
    best_params_gbm = grid_search_gbm.best_params_
    print("Best Params:", best_params_gbm)

    y_pred = best_model_gbm.predict(X_test)
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    return metrics