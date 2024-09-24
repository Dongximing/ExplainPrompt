import pickle


with open("45000_45003_discretize_perturbed_inferenced_df.pkl", "rb") as f:
    reconstructed_df = pickle.load(f)
print(reconstructed_df)