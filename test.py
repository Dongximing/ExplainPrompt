import pickle


with open("1100_1103_discretize_qa_inferenced_df.pkl", "rb") as f:
    reconstructed_df = pickle.load(f)
print(reconstructed_df)