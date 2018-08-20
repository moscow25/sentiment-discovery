import os
import numpy as np
import torch

def main(model_filename, numpy_vec_filename, texts_filename=None, embeddings_filename=None):
    with torch.no_grad():
        model = torch.load(model_filename)['model']
        model.eval()

        # Load a single vector        
        text_vec = np.load(numpy_vec_filename)
        print(text_vec.shape)
        text_vec = torch.from_numpy(text_vec)
        text_vec = text_vec.cuda(non_blocking=True)
        text_vec = torch.autograd.Variable(text_vec)

        glow_vec, _ = model(text_vec)
        text_vec1 = model.infer(glow_vec)
        print(text_vec)
        print(glow_vec)
        print(text_vec1)

        # Now load multiple vectors from a file
        print('loading texts from %s' % texts_filename)
        texts = np.load(texts_filename)
        print('loading embeddings from %s' % embeddings_filename)
        embs = np.load(embeddings_filename)
        
        print(texts)
        glow_embs = np.empty((len(texts), 4096), dtype='float32')
        for i,text in enumerate(list(texts)):
            print('---------------')
            print(text)
            text_vec = embs[i]
            text_vec = np.expand_dims(text_vec, axis=0)
            print(text_vec.shape)
            text_vec = torch.from_numpy(text_vec)
            text_vec = text_vec.cuda(non_blocking=True)
            text_vec = torch.autograd.Variable(text_vec)

            glow_vec, _ = model(text_vec)
            text_vec1 = model.infer(glow_vec)
            print(text_vec)
            print(glow_vec)
            print(text_vec1)

            # Save the GLOW vector
            glow_embs[i] = glow_vec.squeeze().data.cpu().numpy()
        print(glow_embs)

        # Now project in GLOW space -- run 100x random A+B/2 and save to file
        # TODO: Make this command line?
        n_samples = 100
        sample_inputs = [0] * n_samples
        glow_averages = np.empty((n_samples, 4096), dtype='float32')
        project_averages = np.empty((n_samples, 4096), dtype='float32')
        for i in range(n_samples):
            ratio = np.random.choice([0.3,0.5,0.7])
            idx_a = np.random.randint(len(texts))
            idx_b = idx_a
            while idx_b == idx_a:
                idx_b = np.random.randint(len(texts))
            print('-----\naveraging: %s, %s, %.3f' % (idx_a, idx_b, ratio))
            sample_inputs[i] = [texts[idx_a], texts[idx_b], ratio]
            glow_ave = ratio * glow_embs[idx_a] + (1 - ratio) * glow_embs[idx_b]
            print(glow_ave.shape)
            glow_averages[i] = glow_ave

            # Now run network to get a mapping for this projection...
            glow_vec = np.expand_dims(glow_ave, axis=0)
            glow_vec = np.expand_dims(glow_vec, axis=2)
            print(glow_vec.shape)
            glow_vec = torch.from_numpy(glow_vec)
            glow_vec = glow_vec.cuda(non_blocking=True)
            glow_vec = torch.autograd.Variable(glow_vec)
            print(glow_vec)

            # glow_vec, _ = model(text_vec)
            text_vec1 = model.infer(glow_vec)
            print(text_vec1)
            
            project_averages[i] = text_vec1.squeeze().data.cpu().numpy()
        print(project_averages)     

        # Save output of the projected GLOW averages
        PATH = '/home/nyakovenko/sentiment-discovery/text_to_speech'
        np.save(PATH + '/' + 'glow_average_inputs_len128', sample_inputs)
        np.save(PATH + '/' + 'glow_average_outputs_len128', glow_averages)
        np.save(PATH + '/' + 'project_average_outputs_len128', project_averages)

if __name__ == "__main__":
    #model_filename = '/home/dcg-adlr-tts-output.cosmos235/glow/glow_nlp/checkpoint_glow_nlp_100000' 
    #model_filename = '/data/nicky/glow/checkpoint_glow_nlp_100000'
    model_filename = '/data/nicky/glow/checkpoint_glow_nlp_250000'
    #numpy_vec_filename = '/home/rprenger/text_to_speech/textvec.npy' 
    numpy_vec_filename = '/home/nyakovenko/sentiment-discovery/text_to_speech/textvec.npy'
    texts_filename = '/home/nyakovenko/sentiment-discovery/text_to_speech/len128_texts.npy'
    embeddings_filename = '/home/nyakovenko/sentiment-discovery/text_to_speech/len128_hiddens.npy'
    main(model_filename, numpy_vec_filename, texts_filename, embeddings_filename)
