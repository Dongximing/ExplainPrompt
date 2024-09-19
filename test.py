import pickle


with open("../45070_45073perturbed_inferenced_df.pkl", "rb") as f:
    reconstructed_df = pickle.load(f)
print(reconstructed_df)