from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE


param_grid_ada = {
    'n_estimators': [10, 50, 100, 150],
    'learning_rate': [0.001, 0.01, 0.1, 1.0],
    'algorithm' : ['SAMME', 'SAMME.R']
}


def trainAda(X_train, X_test, y_train, y_test):

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    decision_tree = DecisionTreeClassifier(random_state=42)

    model = AdaBoostClassifier(estimator=decision_tree, random_state=42)
    model.fit(X_train, y_train)

    print("Model Parameters:", model.get_params())


    y_pred = model.predict(X_test)

    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    return metrics


def trainAdaGrid(X_train, X_test, y_train, y_test):

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    best_decision_tree = DecisionTreeClassifier(min_samples_leaf=4, min_samples_split=10, criterion='entropy', random_state=42)
    ada_model = AdaBoostClassifier(estimator=best_decision_tree, random_state=42)
    
    grid_search_ada = GridSearchCV(estimator=ada_model, param_grid=param_grid_ada, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search_ada.fit(X_train, y_train)

    best_ada_model = grid_search_ada.best_estimator_
    y_pred = best_ada_model.predict(X_test)

    best_params_ada = grid_search_ada.best_params_
    print("Best Parameters:", best_params_ada)

    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    return metrics 



def trainAdaSMOTE(X_train, X_test, y_train, y_test):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    smote = SMOTE(random_state=1)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


    decision_tree = DecisionTreeClassifier(min_samples_leaf=4, min_samples_split=10, criterion='entropy', random_state=42)
    model = AdaBoostClassifier(estimator=decision_tree, random_state=42, learning_rate=0.1, n_estimators=100)
    model.fit(X_resampled, y_resampled)
    print("Model Parameters:", model.get_params())


    y_pred = model.predict(X_test)
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    return metrics

def trainAdaSMOTEGrid(X_train, X_test, y_train, y_test):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    smote = SMOTE(random_state=1)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    decision_tree = DecisionTreeClassifier(min_samples_leaf=4, min_samples_split=10, criterion='entropy', random_state=42)
    ada_model = AdaBoostClassifier(estimator=decision_tree, random_state=42)

    grid_search_ada = GridSearchCV(estimator=ada_model, param_grid=param_grid_ada, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search_ada.fit(X_resampled, y_resampled)

    best_ada_model = grid_search_ada.best_estimator_
    best_params_ada = grid_search_ada.best_params_
    print("Best Parameters:", best_params_ada)

    y_pred = best_ada_model.predict(X_test)

    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))

    return metrics