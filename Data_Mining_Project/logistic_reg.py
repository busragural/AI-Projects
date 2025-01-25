from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE


param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
    'penalty':['none', 'elasticnet', 'l1', 'l2'], 
    'max_iter' : [50, 100, 1000, 2500, 5000],
    'tol' : [0.00001, 0.0001, 0.001, 0.1]
}

def trainLR(X_train, X_test, y_train, y_test):

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    model = LogisticRegression(random_state=42, max_iter=1000)

    model.fit(X_train, y_train)
    print("Model Parameters:", model.get_params())

    y_pred = model.predict(X_test)

    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    return metrics


def trainLRGrid(X_train, X_test, y_train, y_test):

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    model = LogisticRegression(random_state=42, max_iter=1000)

    grid_search_lr = GridSearchCV(estimator=model, param_grid=param_grid_lr, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

    grid_search_lr.fit(X_train, y_train)

    best_lr_model = grid_search_lr.best_estimator_
    best_params_lr = grid_search_lr.best_params_
    print("Best Parameters:", best_params_lr)
    
    y_pred = best_lr_model.predict(X_test)

    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    return metrics


def trainLRSMOTE(X_train, X_test, y_train, y_test):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    smote = SMOTE(random_state=1)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_resampled, y_resampled)
    print("Model Parameters:", model.get_params())


    y_pred = model.predict(X_test)
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    return metrics

def trainLRSMOTEGrid(X_train, X_test, y_train, y_test):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    smote = SMOTE(random_state=1)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = LogisticRegression(random_state=42, max_iter=1000)
    grid_search_lr = GridSearchCV(estimator=model, param_grid=param_grid_lr, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

    grid_search_lr.fit(X_resampled, y_resampled)

    best_lr_model = grid_search_lr.best_estimator_
    best_params_lr = grid_search_lr.best_params_
    print("Best Parameters:", best_params_lr)
    
    y_pred = best_lr_model.predict(X_test)
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    return metrics