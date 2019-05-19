from tkinter import *
from tkinter import filedialog
import keras, keras.layers as L, keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
import h5py

def build_autoencoder(img_shape, code_size):
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Conv2D(filters=16, kernel_size=3, padding="same", activation="elu"))
    encoder.add(L.Conv2D(filters=32, kernel_size=3, padding="same", activation="elu"))
    encoder.add(L.Conv2D(filters=64, kernel_size=3, padding="same", activation="elu"))
    encoder.add(L.Flatten())
    encoder.add(L.Dense(code_size))
    
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(np.prod((32, 32, 64))))
    decoder.add(L.Reshape((32, 32, 64)))
    decoder.add(L.Conv2DTranspose(filters=32, kernel_size=3, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=16, kernel_size=3, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(3, kernel_size=3, padding='same'))
    
    return encoder, decoder
    
def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))
    
def mid(img_path1, img_path2):
    im1 = image.load_img(img_path1, target_size=(32, 32))
    im2 = image.load_img(img_path2, target_size=(32, 32))
    image1 = image.img_to_array(im1)
    image1 /= 255
    image1 -= 0.5
    image2 = image.img_to_array(im2)
    image2 /= 255
    image2 -= 0.5
    code1, code2 = encoder.predict(np.stack([image1, image2]))
    plt.subplot(1, 7, 1)
    plt.axis("off")
    show_image(image1)
    for a in range(1, 7):
        output_code = code1 * (1 - a / 7) + code2 * a / 7
        output_image = decoder.predict(output_code[None])[0]
        plt.subplot(1, 7, a + 1)
        plt.axis("off")
        show_image(output_image) 
    plt.subplot(1, 7, 7)
    plt.axis("off")
    show_image(image2)
    plt.show()
	
def get_name(path):
	ind = -1
	for i in range(len(path)):
		if path[i] == '/':
			ind = i
	return path[ind + 1:]

def onClick1():
	global dims
	dir_path = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = dims)
	if dir_path != "":
		global path1
		global l1
		path1 = dir_path
		l1.configure(text=get_name(path1))
		
def onClick2():
	global dims
	dir_path = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = dims)
	if dir_path != "":
		global path2
		global l2
		path2 = dir_path
		l2.configure(text=get_name(path2))
	
def onClick3():
	global path1
	global path2
	if path1 != "" and path2 != "":
		mid(path1, path2)
 
width_lab = 32
height_lab = 1   
_bwidth = 30
_bheight = 1
bg_lab = "LightSkyBlue1"
bg_butt = "SteelBlue1"
_font = "Times 15 bold"
dims = (("jpeg files", "*.jpeg"), ("jpg files", "*.jpg"), ("png files", ".png"), ("bmp files", ".bmp"), ("gif files", ".gif"), ("tiff files", ".tiff"))

encoder, decoder = build_autoencoder((32, 32, 3), code_size=32)
encoder.load_weights("encoder.h5")
decoder.load_weights("decoder.h5")
encoder.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
decoder.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
path1 = ""
path2 = ""
root = Tk()
root.title("Средние между картинками")
x = (root.winfo_screenwidth() - root.winfo_reqwidth()) / 2
y = (root.winfo_screenheight() - root.winfo_reqheight()) / 2
root.wm_geometry("+%d+%d" % (x, y))
b1 = Button(root, width = _bwidth, height = _bheight, bg = bg_butt, font = _font, text="Выберите первую картинку", command=onClick1)
l1 = Label(root, width = width_lab, height = height_lab, bg = bg_lab, font = _font)
b2 = Button(root, width = _bwidth, height = _bheight, bg = bg_butt, font = _font, text="Выберите вторую картинку", command=onClick2)
l2 = Label(root, width = width_lab, height = height_lab, bg = bg_lab, font = _font)
b3 = Button(root, width = _bwidth, height = _bheight, bg = bg_butt, font = _font, text="Показать средние", command=onClick3)
b1.pack(fill = 'both', expand = 'yes')
l1.pack(fill = 'both', expand = 'yes')
b2.pack(fill = 'both', expand = 'yes')
l2.pack(fill = 'both', expand = 'yes')
b3.pack(fill = 'both', expand = 'yes')
root.mainloop()
