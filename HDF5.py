import os
import h5py
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
 
images_path = '../data/train'
hdf5_file = '../data/train.h5py'
 
with h5py.File(hdf5_file, 'w', rdcc_nslots=11213, rdcc_nbytes=1024**3, rdcc_w0=1) as hf:
     for idx, image_path in enumerate(glob(os.path.join(images_path, '*'))):
        image = Image.open(image_path)
        
        """
        image 전처리
        """
    
        iset = hf.create_dataset(f'{idx}/image',
                        data=image,
                        shape=(image.height, image.width, 3), # Height, Width, Channels
                        compression='gzip',
                        compression_opts=9,
                        chunks=True)
                        
        """
        Label도 똑같이 적용 가능
        """
        
 # Check !
with h5py.File(hdf5_file, 'r') as hf:
    plt.imshow(hf["0"]["i"])