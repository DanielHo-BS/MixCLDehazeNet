import argparse
import clip
import numpy as np
import os
import PIL
import torch
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


def zero_shout(gpus, images, classTest, batch_size=8, modelName='ViT-B/32'):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    # Load the model
    print('loading model: ', modelName)
    model, preprocess = clip.load(modelName, device)
    
    # Encode the images and text
    image_inputs = torch.stack([preprocess(image) for image in tqdm(images, desc="Processing images")]).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a {c.split(' ')[1]} photo of {c.split(' ')[0]}") for c in classTest]).to(device)

    # Zero shot classification
    similarity = torch.zeros(len(image_inputs), len(text_inputs)).to(device)
    for i in tqdm(range(0, len(image_inputs), batch_size), desc="Zero-shot classification"):
        image_inputs_batch = image_inputs[i:i+batch_size]
        with torch.no_grad():
            image_features_batch = model.encode_image(image_inputs_batch)
            text_features = model.encode_text(text_inputs)
            image_features_batch /= image_features_batch.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity_batch = (100.0 * image_features_batch @ text_features.T).softmax(dim=-1)
            similarity[i:i+batch_size] = similarity_batch

    return {'similarity': similarity, 'text_features': text_features}


def save_text_feature(result, image_paths, save=False):
    values = 0
    for i in tqdm(range(len(image_paths)), desc="Saving text features"):
        value, index = result['similarity'][i].topk(1)
        values += value.item()
        # save text feature for each image
        if save:
            text_feature = result['text_features'][index].cpu().numpy()
            save_path = image_paths[i].replace('.jpg', '.npy').replace('.png', '.npy').replace('hazy', 'text_feature')
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            np.save(save_path, text_feature)

    return values / len(image_paths)

def plot_result(result, images, classTest):
    # plot the images with title of the predicted class
    fig, ax = plt.subplots(2, 5, figsize=(20, 8))
    for x in range(5):
        offset = 100 * x 
        for i in range(10):
            value, index = result['similarity'][offset+i].topk(1)
            ax[i//5, i%5].imshow(images[offset+i])
            ax[i//5, i%5].set_title(f"Predicted: {classTest[index]}")
            ax[i//5, i%5].axis('off')
        # save the plot
        plt.tight_layout()
        plt.savefig('result'+str(x)+'.png')

def main(args):
    # Prepare the inputs
    image_folder_train = os.path.join(args.data_dir, args.dataset, 'train', 'hazy')
    image_folder_test = os.path.join(args.data_dir, args.dataset, 'test', 'hazy')
    image_paths=[]
    for image in os.listdir(image_folder_train):
        if image.endswith('.jpg') or image.endswith('.png'):
            image_paths.append(os.path.join(image_folder_train, image))
    for image in os.listdir(image_folder_test):
        if image.endswith('.jpg') or image.endswith('.png'):
            image_paths.append(os.path.join(image_folder_test, image))
    
    images = [PIL.Image.open(path) for path in image_paths]
    classTest = ['indoor', 'outdoor']
    # classTest = ['indoor clear', 'indoor hazy', 'indoor cloudy',
    #               'outdoor clear', 'outdoor hazy', 'outdoor cloudy']
    
    # Print number of images with hazy
    print('number of images with hazy: ', len(images))
    # Print classes
    print('classes: ', classTest)

    # Run the model for zero-shot classification
    print('--------start zero-shot classification--------')
    time_start = time.time()
    result = zero_shout(args.gpus, images, classTest, modelName=args.model)
    time_end = time.time()
    print('time cost: ', time_end - time_start)
    print('--------end zero-shot classification--------')

    # Save text feature
    average = save_text_feature(result, image_paths, save=args.save)
    print('average similarity: ', average)

    # Plot the result
    # plot_result(result, images, classTest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../datasets/data', type=str, help='path to dataset')
    parser.add_argument('--dataset', default='RESIDE-6K', type=str, help='dataset name')
    parser.add_argument('--gpus', default='0', type=str, help='GPUs used for training')
    parser.add_argument('--model', default='ViT-B/32', type=str, help='model name')
    parser.add_argument('--save', default=False, type=bool, help='save text feature for each image')

    args = parser.parse_args()
    main(args)
