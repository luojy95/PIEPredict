conda create --no-default-packages -n cs598mav python=3.7
conda remove --name cs598mav --all
conda env list # all envs and their locations

conda activate cs598mav
(cs598mav) C:\Users\37845\Anaconda3\envs\cs598mav\Scripts\pip.exe install xx # by calling this specific pip, it will install any non-existing package to the local conda virtualenv! This will not mess up with the base env or other envs. you can check pip list -v for its location
conda deactivate

C:\Users\37845\Anaconda3\envs\cs598mav\Scripts\pip.exe install ./pedestrian-detection-ssdlite

# in PIE folder
# use extract_images_from_videos.py to extract frames

# in PIEPredict folder
# train_test.py (34): dim_ordering = K.image_dim_ordering() --> K.tensorflow_backend.image_dim_ordering()
# pie_intent.py (53): K.set_image_dim_ordering('tf') --> K.tensorflow_backend.set_image_dim_ordering('tf')

# train_test.py (36): set environ
# pie_intent.py (56): set environ

# pie_data.py (92): 'test': ['set03'] --> 'test': ['set02']
# pie_intent.py (244): '/' --> '\\'
