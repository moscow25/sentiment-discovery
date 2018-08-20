import os
import numpy as np
import torch

def main(model_filename, numpy_vec_filename):
    with torch.no_grad():
        model = torch.load(model_filename)['model']
        model.eval()
        
        text_vec = np.load(numpy_vec_filename)
        text_vec = torch.from_numpy(text_vec)
        text_vec = text_vec.cuda(non_blocking=True)
        text_vec = torch.autograd.Variable(text_vec)

        glow_vec, _ = model(text_vec)
        text_vec1 = model.infer(glow_vec)
        print(text_vec)
        print(glow_vec)
        print(text_vec1)
if __name__ == "__main__":
    #model_filename = '/home/dcg-adlr-tts-output.cosmos235/glow/glow_nlp/checkpoint_glow_nlp_100000' 
    model_filename = '/data/nicky/glow/checkpoint_glow_nlp_100000'
    #numpy_vec_filename = '/home/rprenger/text_to_speech/textvec.npy' 
    numpy_vec_filename = '/home/nyakovenko/sentiment-discovery/text_to_speech/textvec.npy'
    main(model_filename, numpy_vec_filename)
