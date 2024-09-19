import pickle


with open("45010_45030perturbed_inferenced_df.pkl", "rb") as f:
    reconstructed_df = pickle.load(f)
print(reconstructed_df)