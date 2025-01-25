from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# Define parameter grid for Extra Trees
param_grid_et = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

def trainET(X_train, X_test, y_train, y_test):

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    model = ExtraTreesClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Model Parameters:", model.get_params())   


    y_pred = model.predict(X_test)

    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    return metrics

def trainETGrid(X_train, X_test, y_train, y_test):

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    et = ExtraTreesClassifier(random_state=42)

    grid_search_et = GridSearchCV(estimator=et, param_grid=param_grid_et, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

    grid_search_et.fit(X_train, y_train)

    best_model_et = grid_search_et.best_estimator_
    best_params_et = grid_search_et.best_params_

    y_pred = best_model_et.predict(X_test)

    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    print("Best Params:", best_params_et)

    return metrics

def trainETSMOTE(X_train, X_test, y_train, y_test):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    smote = SMOTE(random_state=1)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = ExtraTreesClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_test)
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    print("Model Parameters:", model.get_params())
    return metrics

def trainETSMOTEGrid(X_train, X_test, y_train, y_test):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    smote = SMOTE(random_state=1)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    et = ExtraTreesClassifier(random_state=42)
    grid_search_et = GridSearchCV(estimator=et, param_grid=param_grid_et, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

    grid_search_et.fit(X_resampled, y_resampled)

    best_model_et = grid_search_et.best_estimator_
    best_params_et = grid_search_et.best_params_
    print("Best Params:", best_params_et)

    y_pred = best_model_et.predict(X_test)
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    return metrics