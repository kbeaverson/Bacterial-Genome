import datetime

def write_rf_results_to_file(file_path: str, K: int, cv: int, n_iter: int, max_depth_low: int, max_depth_high: int, n_estimators_low: int, n_estimators_high: int, ba_ci: tuple, runtime: datetime):
    with open(file_path, "a") as file: 
        file.write(f"Number of outer folds: {K}\n")
        file.write(f"Number of inner folds: {cv}\n")
        file.write(f"Number of iterations: {n_iter}\n")
        file.write(f"Min tree height: {max_depth_low}\n")
        file.write(f"Max tree height: {max_depth_high}\n")
        file.write(f"Min estimators: {n_estimators_low}\n")
        file.write(f"Max estimators: {n_estimators_high}\n")
        file.write(f"Balanced accuracy confidence interval: {ba_ci}\n")
        file.write(f"Runtime: {runtime}\n\n")
        file.write("========\n\n")

    print("Model performance recorded to output file.")