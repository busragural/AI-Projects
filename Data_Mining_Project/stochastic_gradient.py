from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

param_grid_sgd = {
    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'max_iter': [1000, 2000, 3000],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'class_weight': [None, 'balanced']
}

def trainSGD(X_train, X_test, y_train, y_test):

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    model = SGDClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Model Parameters:", model.get_params())

    y_pred = model.predict(X_test)

    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    return metrics

def trainSGDGrid(X_train, X_test, y_train, y_test):

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    sgd = SGDClassifier(random_state=42)

    grid_search_sgd = GridSearchCV(estimator=sgd, param_grid=param_grid_sgd, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

    grid_search_sgd.fit(X_train, y_train)

    best_model_sgd = grid_search_sgd.best_estimator_
    best_params_sgd = grid_search_sgd.best_params_

    y_pred = best_model_sgd.predict(X_test)

    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))


    print("Best Params:", best_params_sgd)

    print("class: ", classification_report(y_test,y_pred))

    return metrics

def trainSGDSMOTE(X_train, X_test, y_train, y_test):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    smote = SMOTE(random_state=1)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = SGDClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)
    print("Model Parameters:", model.get_params())


    y_pred = model.predict(X_test)
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    return metrics

def trainSGDSMOTEGrid(X_train, X_test, y_train, y_test):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    smote = SMOTE(random_state=1)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    sgd = SGDClassifier(random_state=42)
    grid_search_sgd = GridSearchCV(estimator=sgd, param_grid=param_grid_sgd, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

    grid_search_sgd.fit(X_resampled, y_resampled)

    best_model_sgd = grid_search_sgd.best_estimator_
    best_params_sgd = grid_search_sgd.best_params_
    print("Best Params:", best_params_sgd)

    y_pred = best_model_sgd.predict(X_test)
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    print("class: ", classification_report(y_test, y_pred))
    return metrics