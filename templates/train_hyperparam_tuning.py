"""
TEMPLATE: Hyperparameter Tuning Script

INSTRUCTIONS FOR DATA SCIENTISTS:
1. Choose your search method (Grid Search, Random Search, Bayesian, or Hyperopt)
2. Choose your data loader (delete the other two)
3. Define your hyperparameter search space
4. Uncomment one search method and delete the others
5. Run and compare results in MLflow UI

This template provides:
- Four hyperparameter tuning approaches
- Nested MLflow runs for easy comparison
- Automatic logging of all trials
- Best model selection and registration
"""
import argparse
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np

from src.utils import get_git_metadata, validate_git_state

warnings.filterwarnings('ignore')


# Enable autologging
mlflow.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=False,  # Don't log every trial model (save space)
    disable=False,
    exclusive=False,
    silent=True
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--target-column", type=str, required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    
    # Tuning parameters
    parser.add_argument("--search-method", type=str, default="random",
                       choices=["grid", "random", "bayesian", "hyperopt"])
    parser.add_argument("--n-trials", type=int, default=20,
                       help="Number of trials for random/bayesian/hyperopt search")
    parser.add_argument("--cv-folds", type=int, default=3,
                       help="Number of cross-validation folds")
    
    # MLflow parameters
    parser.add_argument("--experiment-name", type=str, default="hyperparam-tuning")
    parser.add_argument("--model-name", type=str, default="best-model")
    parser.add_argument("--strict-git", type=str, default="false")
    
    return parser.parse_args()


def load_data(args):
    """
    Load data using appropriate data loader.
    Choose ONE, delete the other two.
    """
    
    # OPTION 1: TABULAR DATA
    from src.data_loaders import TabularDataLoader
    
    loader = TabularDataLoader(
        data_path=args.data_path,
        target_column=args.target_column,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state
    )
    
    X_train, X_test, y_train, y_test, X_val, y_val = loader.load_and_split()
    
    # OPTION 2: IMAGE DATA
    # from src.data_loaders import ImageDataLoader
    # loader = ImageDataLoader(...)
    # X_train, X_test, y_train, y_test, X_val, y_val = loader.load_and_split()
    
    # OPTION 3: DATABASE
    # from src.data_loaders import DatabaseDataLoader
    # loader = DatabaseDataLoader(...)
    # X_train, X_test, y_train, y_test, X_val, y_val = loader.load_and_split()
    
    return X_train, X_test, y_train, y_test, X_val, y_val, loader


def define_search_space():
    """
    Define hyperparameter search space.
    
    INSTRUCTIONS:
    Customize this for your model and parameters.
    """
    
    # TODO: Define YOUR hyperparameter search space
    # Example for RandomForest:
    param_space = {
        'n_estimators': [50, 100, 150, 200, 300],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
    }
    
    # Example for XGBoost:
    # param_space = {
    #     'learning_rate': [0.01, 0.05, 0.1, 0.2],
    #     'n_estimators': [50, 100, 200, 300],
    #     'max_depth': [3, 5, 7, 9],
    #     'subsample': [0.6, 0.8, 1.0],
    #     'colsample_bytree': [0.6, 0.8, 1.0],
    # }
    
    # Example for Neural Network:
    # param_space = {
    #     'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    #     'learning_rate_init': [0.001, 0.01, 0.1],
    #     'alpha': [0.0001, 0.001, 0.01],
    #     'batch_size': [32, 64, 128],
    # }
    
    return param_space


def grid_search_tuning(X_train, y_train, X_val, y_val, param_space, args):
    """Grid Search: Try all combinations."""
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    print("\n" + "="*60)
    print("GRID SEARCH HYPERPARAMETER TUNING")
    print("="*60)
    
    # TODO: Replace with your model
    base_model = RandomForestClassifier(random_state=args.random_state)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_space,
        cv=args.cv_folds,
        scoring='accuracy',  # TODO: Change metric if needed
        n_jobs=-1,
        verbose=2
    )
    
    print(f"\nTrying {len(grid_search.cv_results_['params'])} combinations...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Best score: {grid_search.best_score_:.4f}")
    print(f"✓ Best params: {grid_search.best_params_}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def random_search_tuning(X_train, y_train, X_val, y_val, param_space, args):
    """Random Search: Random sampling from space."""
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from scipy.stats import randint, uniform
    
    print("\n" + "="*60)
    print("RANDOM SEARCH HYPERPARAMETER TUNING")
    print("="*60)
    
    # TODO: Replace with your model
    base_model = RandomForestClassifier(random_state=args.random_state)
    
    # Convert lists to distributions for random sampling
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 15),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
    }
    
    # Random search
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=args.n_trials,
        cv=args.cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        random_state=args.random_state,
        verbose=2
    )
    
    print(f"\nTrying {args.n_trials} random combinations...")
    random_search.fit(X_train, y_train)
    
    print(f"\n✓ Best score: {random_search.best_score_:.4f}")
    print(f"✓ Best params: {random_search.best_params_}")
    
    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_


def bayesian_optimization_tuning(X_train, y_train, X_val, y_val, param_space, args):
    """Bayesian Optimization: Smart search using gaussian processes."""
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    from sklearn.ensemble import RandomForestClassifier
    
    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION HYPERPARAMETER TUNING")
    print("="*60)
    
    # TODO: Replace with your model
    base_model = RandomForestClassifier(random_state=args.random_state)
    
    # Define search space for Bayesian optimization
    search_space = {
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(3, 15),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 5),
    }
    
    # Bayesian search
    bayes_search = BayesSearchCV(
        estimator=base_model,
        search_spaces=search_space,
        n_iter=args.n_trials,
        cv=args.cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        random_state=args.random_state,
        verbose=2
    )
    
    print(f"\nRunning {args.n_trials} Bayesian optimization trials...")
    bayes_search.fit(X_train, y_train)
    
    print(f"\n✓ Best score: {bayes_search.best_score_:.4f}")
    print(f"✓ Best params: {bayes_search.best_params_}")
    
    return bayes_search.best_estimator_, bayes_search.best_params_, bayes_search.best_score_


def hyperopt_tuning(X_train, y_train, X_val, y_val, param_space, args):
    """Hyperopt: Advanced hyperparameter optimization."""
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    print("\n" + "="*60)
    print("HYPEROPT HYPERPARAMETER TUNING")
    print("="*60)
    
    # Define search space for Hyperopt
    space = {
        'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200, 300]),
        'max_depth': hp.choice('max_depth', [3, 5, 7, 10, 15]),
        'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4]),
    }
    
    # Objective function
    def objective(params):
        # TODO: Replace with your model
        model = RandomForestClassifier(
            **params,
            random_state=args.random_state
        )
        
        # Cross-validation score
        score = cross_val_score(
            model, X_train, y_train,
            cv=args.cv_folds,
            scoring='accuracy'
        ).mean()
        
        # Hyperopt minimizes, so return negative score
        return {'loss': -score, 'status': STATUS_OK}
    
    # Run optimization
    trials = Trials()
    print(f"\nRunning {args.n_trials} Hyperopt trials...")
    
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=args.n_trials,
        trials=trials,
        rstate=np.random.default_rng(args.random_state)
    )
    
    # Train final model with best params
    best_params = {
        'n_estimators': [50, 100, 150, 200, 300][best['n_estimators']],
        'max_depth': [3, 5, 7, 10, 15][best['max_depth']],
        'min_samples_split': [2, 5, 10][best['min_samples_split']],
        'min_samples_leaf': [1, 2, 4][best['min_samples_leaf']],
    }
    
    best_model = RandomForestClassifier(**best_params, random_state=args.random_state)
    best_model.fit(X_train, y_train)
    best_score = best_model.score(X_val, y_val) if X_val is not None else -min([t['loss'] for t in trials.trials])
    
    print(f"\n✓ Best score: {best_score:.4f}")
    print(f"✓ Best params: {best_params}")
    
    return best_model, best_params, best_score


def main():
    """Main hyperparameter tuning pipeline."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING PIPELINE")
    print("="*60)
    
    # Git metadata
    git_metadata = get_git_metadata()
    validate_git_state(git_metadata, strict=args.strict_git.lower() == "true")
    
    # MLflow setup
    mlflow.set_experiment(args.experiment_name)
    
    # Parent run for all tuning trials
    with mlflow.start_run(run_name=f"tuning_{args.search_method}") as parent_run:
        print(f"\n✓ Parent MLflow Run ID: {parent_run.info.run_id}")
        
        # Log git metadata
        for key, value in git_metadata.items():
            mlflow.set_tag(key, value)
        
        mlflow.set_tag("tuning_method", args.search_method)
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param("cv_folds", args.cv_folds)
        
        # Load data
        print("\n" + "-"*60)
        print("Loading data...")
        X_train, X_test, y_train, y_test, X_val, y_val, loader = load_data(args)
        
        print(f"✓ Train: {len(X_train)} samples")
        print(f"✓ Test: {len(X_test)} samples")
        if X_val is not None:
            print(f"✓ Validation: {len(X_val)} samples")
        
        print(loader.summary())
        
        # Define search space
        param_space = define_search_space()
        print(f"\n✓ Search space defined: {len(param_space)} parameters")
        
        # Run hyperparameter tuning
        print("\n" + "-"*60)
        
        if args.search_method == "grid":
            best_model, best_params, best_score = grid_search_tuning(
                X_train, y_train, X_val, y_val, param_space, args
            )
        elif args.search_method == "random":
            best_model, best_params, best_score = random_search_tuning(
                X_train, y_train, X_val, y_val, param_space, args
            )
        elif args.search_method == "bayesian":
            best_model, best_params, best_score = bayesian_optimization_tuning(
                X_train, y_train, X_val, y_val, param_space, args
            )
        elif args.search_method == "hyperopt":
            best_model, best_params, best_score = hyperopt_tuning(
                X_train, y_train, X_val, y_val, param_space, args
            )
        
        # Log best parameters
        for param, value in best_params.items():
            mlflow.log_param(f"best_{param}", value)
        
        mlflow.log_metric("best_cv_score", best_score)
        
        # Evaluate on test set
        print("\n" + "-"*60)
        print("Evaluating best model on test set...")
        test_score = best_model.score(X_test, y_test)
        mlflow.log_metric("test_score", test_score)
        print(f"✓ Test score: {test_score:.4f}")
        
        # Save and log best model
        print("\n" + "-"*60)
        print("Saving best model...")
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        import pickle
        model_path = models_dir / "best_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        # Log model to MLflow
        from mlflow.models import infer_signature
        signature = infer_signature(X_train, best_model.predict(X_train))
        
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            signature=signature,
            registered_model_name=args.model_name
        )
        
        print(f"✓ Best model registered as '{args.model_name}'")
        
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING COMPLETE")
        print("="*60)
        print(f"\nBest parameters: {best_params}")
        print(f"Best CV score: {best_score:.4f}")
        print(f"Test score: {test_score:.4f}")
        print(f"\nParent Run ID: {parent_run.info.run_id}")
        print("\nView all trials in MLflow UI:")
        print("  mlflow ui --port 5000")
        print(f"  Filter by parent run: {parent_run.info.run_id}")
        
        return parent_run.info.run_id


if __name__ == "__main__":
    main()
