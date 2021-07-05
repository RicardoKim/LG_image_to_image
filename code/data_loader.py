from tensorflow.keras.utils import Sequence
import math
import cv2
import tensorflow as tf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
Root_Path = "../"

class Dataloader(Sequence):

    def __init__(self, x_list, y_list, batch_size, shuffle=False):
        self.x, self.y = x_list, y_list
        self.batch_size = batch_size
        self.img_size = 256
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_x = []
        batch_y = []
        
        for i in indices:
            print("=", end = ' ')
            x_image = cv2.imread(self.x[i])
            y_image = cv2.imread(self.y[i])

            top = np.random.randint(x_image.shape[0])
            left = np.random.randint(x_image.shape[1])
            x_piece = np.zeros([self.img_size, self.img_size, 3], np.uint8)
            x_temp = x_image[top : top + self.img_size, left : left + self.img_size, :]
            x_piece[:x_temp.shape[0], :x_temp.shape[1], :] = x_temp

            y_piece = np.zeros([self.img_size, self.img_size, 3], np.uint8)
            y_temp = y_image[top : top + self.img_size, left : left + self.img_size, :]
            y_piece[:y_temp.shape[0], :y_temp.shape[1], :] = y_temp

            batch_x.append(x_piece)
            batch_y.append(y_piece)
        
        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indices)


def preparing():
    train_csv = pd.read_csv(Root_Path + "train.csv")
    train_all_input_files = Root_Path + 'raw_data/train_input_img/'+train_csv['input_img']
    train_all_label_files = Root_Path + 'raw_data/train_label_img/'+train_csv['label_img']
    train_input_files = train_all_input_files[60:].to_numpy()
    train_label_files = train_all_label_files[60:].to_numpy()
    train_loader = Dataloader(train_input_files, train_label_files, 32, shuffle=True)

    return train_loader

def main():
    train_loader = preparing()
    for e in range(1):
        for x, y in train_loader:
            plt.figure(figsize=(15,10))
            inp_img = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            targ_img = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
            plt.subplot(1,2,1)
            plt.imshow(inp_img)
            plt.subplot(1,2,2)
            plt.imshow(targ_img)
            plt.show()


if __name__ == "__main__":
    main()