import pickle

def save_model(model, filename):
    pickle_out = open(filename, "wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()
